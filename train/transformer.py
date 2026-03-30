import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import os

# =============================================================================
# 超参数 —— 极小模型 + 强正则
# =============================================================================
INPUT_DIM = 624       # 加 Δforce 后的维度
SEQ_LEN = 20          # 20帧一组
D_MODEL = 32          # 大幅缩小
N_HEAD = 2            # 2个注意力头
NUM_LAYERS = 1        # 1层Encoder
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
# 🔥 最终超轻量 Transformer —— 每组20帧输出一个标签
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
        
        # 输出层（20帧 → 1个输出）
        self.norm = nn.LayerNorm(D_MODEL)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(D_MODEL, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = (B, 20, 624)
        
        x = self.proj(x)
        x = x + self.pos_emb
        
        # 时序特征提取
        x = self.transformer(x)
        
        # 关键：20帧全局池化 → 整段序列只输出1个结果！
        x = x.mean(dim=1)
        
        x = self.norm(x)
        x = self.drop(x)
        
        # 最终输出：每组序列一个概率（滑/稳）
        out = self.sigmoid(self.fc(x))
        return out

# =============================================================================
# 训练 & 评估
# =============================================================================
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience=8):
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
            torch.save(model.state_dict(), "/home/liuli/tactile_lstm/models/transformer_all.pth")
            counter = 0
            print("✅ 保存最佳模型")
        else:
            counter += 1
            if counter >= patience:
                print("🛑 早停")
                break

# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    data_folder = "/home/liuli/tactile_lstm/train_data/data_all"
    all_files = os.listdir(data_folder)

    train_files = [f for f in all_files if f.endswith(".npz") and not f.startswith("val_")]
    val_files = [f for f in all_files if f.endswith(".npz") and f.startswith("val_")]

    X_train, y_train = load_npz_list(data_folder, train_files)
    X_val, y_val = load_npz_list(data_folder, val_files)

    # ======================
    # 🔥 这里显示训练集 / 验证集数量
    # ======================
    print(f"训练集数量: {X_train.shape[0]} 组序列")
    print(f"验证集数量: {X_val.shape[0]} 组序列")
    print(f"训练集滑/稳标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"验证集滑/稳标签分布: {np.unique(y_val, return_counts=True)}")

    # 加入Δforce
    X_train = add_delta_feature(X_train)
    X_val = add_delta_feature(X_val)

    # 数据集
    train_dataset = TactileDataset(X_train, y_train)
    val_dataset = TactileDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # 模型
    model = TinyTactileTransformer().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    print("\n🚀 超轻量Transformer训练开始（每组序列输出一个标签）\n")
    train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, DEVICE, EPOCHS)
