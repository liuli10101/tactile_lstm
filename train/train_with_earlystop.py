import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import os


# =====================================================
# 1 加载指定文件列表（无修改）
# =====================================================
def load_npz_list(folder_path, file_list):
    X_list = []
    y_list = []

    for file in file_list:
        full_path = os.path.join(folder_path, file)
        print("加载:", file)
        data = np.load(full_path)
        X = data["X"]
        y = data["y"]
        print("  X shape:", X.shape)
        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    print("合并后 shape:", X_all.shape)
    return X_all, y_all


# =====================================================
# 2 Dataset（无修改）
# =====================================================
class TactileDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================================================
# 3 模型（无修改）
# =====================================================
class SlipDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_proj = nn.Linear(312, 128)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_proj(x)
        x = self.relu(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze()


# =====================================================
# 4 训练（修正：返回最优模型 + 最优验证精度）
# =====================================================
def train_model(model, train_loader, val_loader, device, epochs=30, patience=5):
    """
    返回：
    - model: 加载了最优权重的模型
    - best_val_acc: 训练过程中的最优验证集精度
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    # 早停相关初始化
    best_val_acc = 0.0
    best_model_weights = None
    counter = 0

    for epoch in range(epochs):
        # ====== Train ======
        model.train()
        total_loss = 0
        train_preds = []
        train_labels = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        # ====== Validation ======
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                preds = (outputs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        # ====== 早停逻辑 ======
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()
            counter = 0
            print(f"→ 验证集精度提升至 {best_val_acc:.4f}，保存最优权重")
        else:
            counter += 1
            print(f"→ 验证集精度未提升，计数器: {counter}/{patience}")

        # 打印日志
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {total_loss:.4f} "
              f"Train Acc: {train_acc:.4f} "
              f"Val Acc: {val_acc:.4f} "
              f"Best Val Acc: {best_val_acc:.4f}")

        # 触发早停
        if counter >= patience:
            print(f"\n早停触发！连续{patience}轮验证集精度未提升")
            break

    # 加载最优权重
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"\n训练完成，加载最优模型权重（最优验证集精度：{best_val_acc:.4f}）")
    else:
        print("\n训练完成，未找到最优权重（可能仅训练了1轮）")

    # 同时返回模型和最优验证精度
    return model, best_val_acc


# =====================================================
# 5 主程序（修正：接收返回的best_val_acc）
# =====================================================
if __name__ == "__main__":
    data_folder = "/home/liuli/tactile_lstm/train_data/data_306311"
    all_files = os.listdir(data_folder)
    train_files = [f for f in all_files if f.endswith(".npz") and not f.startswith("val_")]
    val_files = [f for f in all_files if f.endswith(".npz") and f.startswith("val_")]

    print("训练文件:", train_files)
    print("验证文件:", val_files)

    X_train, y_train = load_npz_list(data_folder, train_files)
    X_val, y_val = load_npz_list(data_folder, val_files)

    # 标准化
    mean = X_train.mean()
    std = X_train.std() + 1e-8
    print("训练集均值:", mean)
    print("训练集标准差:", std)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # 打印类别分布
    print("\n训练集类别分布:")
    print(np.unique(y_train, return_counts=True))
    print("验证集类别分布:")
    print(np.unique(y_val, return_counts=True))

    # Dataset & DataLoader
    train_dataset = TactileDataset(X_train, y_train)
    val_dataset = TactileDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    model = SlipDetectionModel()

    # ====== 修正：接收返回的模型和最优验证精度 ======
    model, best_val_acc = train_model(model, train_loader, val_loader, device, epochs=30, patience=5)

    # 保存最优模型（含均值/标准差+最优精度）
    torch.save({
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "best_val_acc": best_val_acc  # 现在可以正常访问
    }, "/home/liuli/tactile_lstm/models/lstm311_best.pth")

    print(f"最优模型已保存（最优验证集精度：{best_val_acc:.4f}）")
