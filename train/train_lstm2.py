import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os



# 1 Δforce 特征

def add_delta_feature(X):

    # X (N,T,312)

    delta = np.diff(X, axis=1)

    delta = np.concatenate(
        [np.zeros((X.shape[0],1,X.shape[2])), delta],
        axis=1
    )

    X_new = np.concatenate([X, delta], axis=2)

    print("加入Δforce后 shape:", X_new.shape)

    return X_new



# 2 加载指定文件

def load_npz_list(folder_path, file_list):

    X_list = []
    y_list = []

    for file in file_list:

        full_path = os.path.join(folder_path, file)

        print("加载:", file)

        data = np.load(full_path)

        X = data["X"]
        y = data["y"]

        print("X shape:", X.shape)

        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    print("合并后:", X_all.shape)

    return X_all, y_all



# 3 Dataset
class TactileDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]



# 4 模型
class SlipDetectionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # 输入维度 624
        self.feature_proj = nn.Linear(624, 128)

        self.relu = nn.ReLU()

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        self.attn = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x (B,T,624)

        x = self.feature_proj(x)
        x = self.relu(x)

        lstm_out, _ = self.lstm(x)

        weights = torch.softmax(self.attn(lstm_out), dim=1)

        context = torch.sum(lstm_out * weights, dim=1)

        out = self.dropout(context)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)

        out = self.sigmoid(out)

        return out.squeeze()



# 5 训练
def train_model(model, train_loader, val_loader, device, epochs=30, patience=5):

    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_f1 = 0
    counter = 0

    model.to(device)

    for epoch in range(epochs):

        # ====================
        # train
        # ====================

        model.train()

        train_preds = []
        train_labels = []

        total_loss = 0

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

        # ====================
        # validation
        # ====================

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

        val_precision = precision_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss {total_loss:.3f} "
            f"TrainAcc {train_acc:.3f} "
            f"ValAcc {val_acc:.3f} "
            f"F1 {val_f1:.3f}"
        )

        # ====================
        # Early stopping
        # ====================

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(model.state_dict(), "/home/liuli/tactile_lstm/models/lstm2_311.pth")

            counter = 0

            print("保存最佳模型")

        else:

            counter += 1

            if counter >= patience:

                print("Early stopping")

                break



# 6 主程序
if __name__ == "__main__":

    data_folder = "/home/liuli/tactile_lstm/train_data/data_306311"

    all_files = os.listdir(data_folder)

    train_files = [
        f for f in all_files
        if f.endswith(".npz") and not f.startswith("val_")
    ]

    val_files = [
        f for f in all_files
        if f.endswith(".npz") and f.startswith("val_")
    ]

    print("train:", train_files)
    print("val:", val_files)

    X_train, y_train = load_npz_list(data_folder, train_files)
    X_val, y_val = load_npz_list(data_folder, val_files)

    # =========================
    # Δforce
    # =========================

    X_train = add_delta_feature(X_train)
    X_val = add_delta_feature(X_val)

    # =========================
    # 标准化
    # =========================

    mean = X_train.mean(axis=(0,1), keepdims=True)
    std = X_train.std(axis=(0,1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    print("train label:", np.unique(y_train, return_counts=True))

    # =========================
    # dataset
    # =========================

    train_dataset = TactileDataset(X_train, y_train)
    val_dataset = TactileDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # =========================
    # device
    # =========================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)

    model = SlipDetectionModel()

    train_model(model, train_loader, val_loader, device, epochs=30)

    # =========================
    # save
    # =========================

    torch.save({

        "model_state_dict": model.state_dict(),

        "mean": mean,

        "std": std

    }, "/home/liuli/tactile_lstm/models/lstm2_311.pth")

    print("模型保存完成")