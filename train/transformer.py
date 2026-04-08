import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import os

# =============================================================================
# 超参数
# =============================================================================
INPUT_DIM = 624       
SEQ_LEN = 20          
D_MODEL = 32          
N_HEAD = 2            
NUM_LAYERS = 1        
BATCH_SIZE = 16
EPOCHS = 30
LR = 5e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 数据读取
# =============================================================================
def load_npz_list(folder_path, file_list):
    X_list = []
    y_list = []
    for file in file_list:
        full_path = os.path.join(folder_path, file)
        data = np.load(full_path)
        X = data["X"]
        y = data["y"]
        X_list.append(X)
        y_list.append(y)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all

def add_delta_feature(X):
    delta = np.diff(X, axis=1)
    delta = np.concatenate([np.zeros((X.shape[0],1,X.shape[2])), delta], axis=1)
    return np.concatenate([X, delta], axis=2)

class TactileDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =============================================================================
# TinyTactileTransformer 模型
# =============================================================================
class TinyTactileTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 输入投影
        self.proj = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL))
        
        # 极小Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=D_MODEL * 2,
            batch_first=True,
            dropout=0.4,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        
        # 输出层
        self.norm = nn.LayerNorm(D_MODEL)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(D_MODEL, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.proj(x)
        x = x + self.pos_emb
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.drop(x)
        out = self.sigmoid(self.fc(x))
        return out

# =============================================================================
# 训练 & 评估
# =============================================================================
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path, patience=8):
    best_acc = 0
    counter = 0

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # 评估
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
                all_preds.extend((pred > 0.2).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(f"[Epoch {epoch+1:2d}] TrainLoss:{train_loss:.4f} | ValLoss:{val_loss:.4f} | Acc:{acc:.3f} | F1:{f1:.3f}")

        if acc > best_acc:
            best_acc = acc
            # 这里不再单独保存 state_dict，而是在最后统一保存
            # torch.save(model.state_dict(), save_path)
            counter = 0
            print("✅ 当前最佳模型")
        else:
            counter += 1
            if counter >= patience:
                print("🛑 早停")
                break
    
    return best_acc

# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    data_folder = "/home/liuli/tactile_lstm/train_data/data_all"
    model_save_path = "/home/liuli/tactile_lstm/models/transformer_all.pth"
    
    all_files = os.listdir(data_folder)

    train_files = [f for f in all_files if f.endswith(".npz") and not f.startswith("val_")]
    val_files = [f for f in all_files if f.endswith(".npz") and f.startswith("val_")]

    X_train, y_train = load_npz_list(data_folder, train_files)
    X_val, y_val = load_npz_list(data_folder, val_files)

    print(f"训练集数量: {X_train.shape[0]} 组序列")
    print(f"验证集数量: {X_val.shape[0]} 组序列")
    print(f"训练集滑/稳标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"验证集滑/稳标签分布: {np.unique(y_val, return_counts=True)}")

    # ======================
    # 🔥 1. 加入Δforce
    # ======================
    X_train = add_delta_feature(X_train)
    X_val = add_delta_feature(X_val)

    # ======================
    # 🔥 2. 【新增】计算并保存训练集的 mean 和 std
    # ======================
    print("\n正在计算训练集标准化参数...")
    # 注意：只在 X_train 上计算，保持 (1, 1, 624) 的维度以便广播
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    
    # 应用标准化
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    print("标准化完成")

    # ======================
    # 数据集
    # ======================
    train_dataset = TactileDataset(X_train, y_train)
    val_dataset = TactileDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # ======================
    # 模型
    # ======================
    model = TinyTactileTransformer().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    print("\n🚀 超轻量Transformer训练开始\n")
    best_acc = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, DEVICE, EPOCHS, model_save_path)

    # ======================
    # 🔥 3. 【修改】保存完整 Checkpoint (包含 model, mean, std)
    # ======================
    print(f"\n💾 正在保存完整模型至: {model_save_path}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "best_acc": best_acc
    }, model_save_path)
    
    print("✅ 模型保存完成！包含 model_state_dict, mean 和 std")
