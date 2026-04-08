import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

# =============================================================================
# 🔥 1. 模型定义 (带时间衰减 Attention，与推理代码完全一致)
# =============================================================================
class TinyTactileTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        INPUT_DIM = 624
        SEQ_LEN = 20
        D_MODEL = 64
        N_HEAD = 2
        NUM_LAYERS = 1
        
        # 输入投影
        self.proj = nn.Linear(INPUT_DIM, D_MODEL)
        self.norm_proj = nn.LayerNorm(D_MODEL)
        self.relu = nn.GELU()
        
        # 位置编码
        self.pos_emb = nn.Parameter(torch.empty(1, SEQ_LEN, D_MODEL))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=D_MODEL * 4,
            batch_first=True,
            dropout=0.3,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        
        # Attention Pooling 层
        self.attn = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.Tanh(),
            nn.Linear(D_MODEL // 2, 1)
        )
        
        self.norm = nn.LayerNorm(D_MODEL)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(D_MODEL, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm_proj(x)
        x = self.relu(x)
        
        x = x + self.pos_emb
        x = self.transformer(x)
        
        # 🔴 核心修改：时间衰减 Attention (与推理代码一致)
        attn_weights = self.attn(x)  # (B, T, 1)
        
        # 生成时间衰减因子：从 0.5 线性增加到 1.0
        time_decay = torch.linspace(0.1, 1.0, steps=x.size(1), device=x.device)
        time_decay = time_decay.view(1, -1, 1)
        
        # 应用衰减并重新归一化
        attn_weights = attn_weights * time_decay
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 加权求和
        x = torch.sum(x * attn_weights, dim=1)
        
        x = self.norm(x)
        x = self.drop(x)
        out = self.sigmoid(self.fc(x))
        return out.squeeze()

    # 保持兼容
    def add_delta_feature(self, window):
        delta = np.diff(window, axis=0)
        delta = np.concatenate([np.zeros((1, window.shape[1])), delta], axis=0)
        window_new = np.concatenate([window, delta], axis=1)
        return window_new

# =============================================================================
# 2. 数据处理部分
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
# 3. 训练循环
# =============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path_base, patience=8):
    best_f1 = 0.0
    counter = 0
    grad_clip = 1.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
                all_preds.extend((pred > 0.5).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(f"[Epoch {epoch+1:2d}] Loss:{train_loss:.4f} | ValLoss:{val_loss:.4f} | Acc:{acc:.3f} | Precision:{precision:.3f} | Recall:{recall:.3f} | F1:{f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            print(f"✅ 保存最佳模型 (Best F1: {best_f1:.3f})")
            
            torch.save(model.state_dict(), save_path_base + ".pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "mean": mean,
                "std": std,
                "best_f1": best_f1
            }, save_path_base + "_full.pth")
            
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("🛑 早停")
                break
    return best_f1

# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    # 固定随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 配置路径
    data_folder = "/home/liuli/tactile_lstm/train_data/data_all"
    model_save_base = "/home/liuli/tactile_lstm/models/transformer2" # 新文件名，区分开
    
    # 读取数据
    all_files = os.listdir(data_folder)
    train_files = [f for f in all_files if f.endswith(".npz") and not f.startswith("val_")]
    val_files = [f for f in all_files if f.endswith(".npz") and f.startswith("val_")]

    X_train, y_train = load_npz_list(data_folder, train_files)
    X_val, y_val = load_npz_list(data_folder, val_files)

    # 1. 加 Delta
    X_train = add_delta_feature(X_train)
    X_val = add_delta_feature(X_val)

    # 2. 计算标准化参数
    print("计算标准化参数...")
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # 3. DataLoader
    train_dataset = TactileDataset(X_train, y_train)
    val_dataset = TactileDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # 4. 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyTactileTransformer().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🚀 模型参数量: {total_params/1e3:.1f}K (带时间衰减)")
    print(f"🚀 开始训练，设备: {device}")
    
    best_f1 = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=30, save_path_base=model_save_base)

    print("\n训练完成！")
    print(f"请将推理代码中的模型路径改为: {model_save_base}_full.pth")
