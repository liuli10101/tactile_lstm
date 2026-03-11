import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import os


# =====================================================
# 1 加载指定文件列表
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
# 2 Dataset
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
# 3 模型
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
# 4 训练
# =====================================================
def train_model(model, train_loader, val_loader, device, epochs=30):

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

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

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {total_loss:.4f} "
              f"Train Acc: {train_acc:.4f} "
              f"Val Acc: {val_acc:.4f}")


# =====================================================
# 5 主程序
# =====================================================
if __name__ == "__main__":

    data_folder = "/home/liuli/tactile_lstm/train_data/data306"

    all_files = os.listdir(data_folder)

    # 训练集 = 不以 val_ 开头
    train_files = [f for f in all_files
                   if f.endswith(".npz") and not f.startswith("val_")]

    # 验证集 = 以 val_ 开头
    val_files = [f for f in all_files
                 if f.endswith(".npz") and f.startswith("val_")]

    print("训练文件:", train_files)
    print("验证文件:", val_files)

    # 加载
    X_train, y_train = load_npz_list(data_folder, train_files)
    X_val, y_val = load_npz_list(data_folder, val_files)

    #标准化
    mean = X_train.mean()
    std = X_train.std() + 1e-8   # 防止除零

    print("训练集均值:", mean)
    print("训练集标准差:", std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # 打印类别分布
    print("\n训练集类别分布:")
    print(np.unique(y_train, return_counts=True))

    print("验证集类别分布:")
    print(np.unique(y_val, return_counts=True))

    # Dataset
    train_dataset = TactileDataset(X_train, y_train)
    val_dataset = TactileDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    model = SlipDetectionModel()

    train_model(model, train_loader, val_loader, device, epochs=15)

    torch.save({
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std
    }, "slip_model.pth")

    print("模型已保存")
