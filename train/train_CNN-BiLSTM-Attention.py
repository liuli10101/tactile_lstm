import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# =====================================================
# 1 数据加载
# =====================================================

def load_npz_list(folder_path, file_list):

    X_list = []
    y_list = []

    for file in file_list:

        full_path = os.path.join(folder_path, file)
        print("加载:", file)

        data = np.load(full_path)

        X = data["X"]      # (N,T,312)
        y = data["y"]

        print("shape:", X.shape)

        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    print("合并后:", X_all.shape)

    return X_all, y_all


# =====================================================
# 2 Dataset
# =====================================================

class TactileDataset(Dataset):

    def __init__(self, X, y, augment=False):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.y[idx]

        # 数据增强
        if self.augment:

            noise = torch.randn_like(x) * 0.01
            x = x + noise

        return x, y


# =====================================================
# 3 Attention模块
# =====================================================

class AttentionPooling(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # x : (B,T,H)

        weights = torch.softmax(self.attn(x), dim=1)

        out = torch.sum(x * weights, dim=1)

        return out


# =====================================================
# 4 模型
# =====================================================

class SlipDetectionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # Temporal CNN
        self.conv1 = nn.Conv1d(312, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        self.attn = AttentionPooling(128)

        # FC
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        # x (B,T,312)

        x = x.permute(0,2,1)  # (B,312,T)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        x = x.permute(0,2,1)  # (B,T,128)

        lstm_out,_ = self.lstm(x)

        attn_out = self.attn(lstm_out)

        out = self.dropout(attn_out)

        out = torch.relu(self.fc1(out))

        out = self.fc2(out)

        return out.squeeze()


# =====================================================
# 5 训练函数
# =====================================================

def train_model(model, train_loader, val_loader, device, epochs=30):

    pos_weight = torch.tensor([3.0]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    for epoch in range(epochs):

        # -----------------------
        # train
        # -----------------------

        model.train()

        train_preds = []
        train_labels = []

        total_loss = 0

        for X_batch,y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs,y_batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            probs = torch.sigmoid(outputs)

            preds = (probs>0.5).float()

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(train_labels,train_preds)

        # -----------------------
        # validation
        # -----------------------

        model.eval()

        val_preds = []
        val_labels = []

        with torch.no_grad():

            for X_batch,y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)

                probs = torch.sigmoid(outputs)

                preds = (probs>0.5).float()

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_acc = accuracy_score(val_labels,val_preds)

        val_precision = precision_score(val_labels,val_preds)
        val_recall = recall_score(val_labels,val_preds)
        val_f1 = f1_score(val_labels,val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss {total_loss:.3f} "
            f"TrainAcc {train_acc:.3f} "
            f"ValAcc {val_acc:.3f} "
            f"F1 {val_f1:.3f}"
        )


# =====================================================
# 6 主程序
# =====================================================

if __name__ == "__main__":

    data_folder = "/home/liuli/tactile_lstm/train_data/data306"

    all_files = os.listdir(data_folder)

    train_files = [f for f in all_files if f.endswith(".npz") and not f.startswith("val_")]

    val_files = [f for f in all_files if f.endswith(".npz") and f.startswith("val_")]

    print("train files:",train_files)
    print("val files:",val_files)

    X_train,y_train = load_npz_list(data_folder,train_files)
    X_val,y_val = load_npz_list(data_folder,val_files)

    # =====================================================
    # 标准化（按特征）
    # =====================================================

    mean = X_train.mean(axis=(0,1),keepdims=True)
    std = X_train.std(axis=(0,1),keepdims=True) + 1e-8

    X_train = (X_train-mean)/std
    X_val = (X_val-mean)/std

    print("train label:",np.unique(y_train,return_counts=True))

    # =====================================================
    # dataset
    # =====================================================

    train_dataset = TactileDataset(X_train,y_train,augment=True)

    val_dataset = TactileDataset(X_val,y_val)

    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

    val_loader = DataLoader(val_dataset,batch_size=32)

    # =====================================================
    # device
    # =====================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:",device)

    # =====================================================
    # model
    # =====================================================

    model = SlipDetectionModel()

    train_model(model,train_loader,val_loader,device,epochs=20)

    # =====================================================
    # save
    # =====================================================

    torch.save({

        "model_state_dict":model.state_dict(),

        "mean":mean,

        "std":std

    },"slip_model_cba.pth")

    print("模型保存完成")