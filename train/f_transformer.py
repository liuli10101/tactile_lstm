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
INPUT_DIM = 625       # 624 + 1维夹紧力
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

# =============================================================================
# 🔥 按你的规则计算夹紧力：第3、6、9...312项求和 / 2
# =============================================================================
def compute_grip_force_per_frame(frame_312):
    # frame_312: (312,)
    # 取索引 2,5,8,...,311 （第3、6、9...312个数）
    indices = np.arange(2, 312, 3)
    sum_val = np.sum(frame_312[indices])
    force = sum_val / 2.0
    return force

def add_force_feature(X):
    # X: (B, 20, 312) 原始触觉数据
    B, T, _ = X.shape
    force_seq = np.zeros((B, T, 1), dtype=np.float32)

    for b in range(B):
        for t in range(T):
            frame = X[b, t]
            f = compute_grip_force_per_frame(frame)
            force_seq[b, t, 0] = f

    # 先做差分特征得到 (B,20,624)，再与力拼接成 (B,20,625)
    x_delta = add_delta_feature(X)
    x_with_force = np.concatenate([x_delta, force_seq], axis=2)
    return x_with_force

class TactileDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =============================================================================
# Transformer 模型（输入 625 维）
# =============================================================================
class TinyTactileTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.proj = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=D_MODEL * 2,
            batch_first=True,
            dropout=0.4,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        
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
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience=8):
    best_acc = 0
    counter = 0

    for epoch in range(EPOCHS):
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
            torch.save(model.state_dict(), "/home/liuli/tactile_lstm/models/f_transformer.pth")
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
    data_folder = "/home/liuli/tactile_lstm/train_data/data226"
    all_files = os.listdir(data_folder)

    train_files = [f for f in all_files if f.endswith(".npz") and not f.startswith("val_")]
    val_files = [f for f in all_files if f.endswith(".npz") and f.startswith("val_")]

    X_train, y_train = load_npz_list(data_folder, train_files)
    X_val, y_val = load_npz_list(data_folder, val_files)

    print(f"训练集数量: {X_train.shape[0]} 组序列")
    print(f"验证集数量: {X_val.shape[0]} 组序列")
    print(f"训练集滑/稳标签分布: {np.unique(y_train, return_counts=True)}")
    print(f"验证集滑/稳标签分布: {np.unique(y_val, return_counts=True)}")

    # 🔥 直接一步生成：差分 + 夹紧力 → (B,20,625)
    X_train = add_force_feature(X_train)
    X_val = add_force_feature(X_val)

    train_dataset = TactileDataset(X_train, y_train)
    val_dataset = TactileDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = TinyTactileTransformer().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    print("\n🚀 带自动夹紧力特征的 Transformer 训练开始\n")
    train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, DEVICE, EPOCHS)
