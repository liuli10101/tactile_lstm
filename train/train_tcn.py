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

        X = data["X"]
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

        if self.augment:

            noise = torch.randn_like(x) * 0.02
            scale = 1 + torch.randn(1) * 0.05

            x = (x + noise) * scale

        return x, y


# =====================================================
# 3 Attention Pooling
# =====================================================

class AttentionPooling(nn.Module):

    def __init__(self, feature_dim):

        super().__init__()

        self.attn = nn.Linear(feature_dim, 1)

    def forward(self, x):

        # x (B,T,F)

        weights = torch.softmax(self.attn(x), dim=1)

        out = torch.sum(x * weights, dim=1)

        return out


# =====================================================
# 4 TCN Block
# =====================================================

class TCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation):

        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = None

    def forward(self, x):

        res = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        if self.residual is not None:
            res = self.residual(res)

        return x + res


# =====================================================
# 5 CNN + TCN + Attention 模型
# =====================================================

class SlipDetectionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # Temporal CNN
        self.conv1 = nn.Conv1d(312, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        # TCN
        self.tcn1 = TCNBlock(128,128,3,dilation=1)
        self.tcn2 = TCNBlock(128,128,3,dilation=2)
        self.tcn3 = TCNBlock(128,128,3,dilation=4)

        # Attention
        self.attn = AttentionPooling(128)

        # FC
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,1)

        self.dropout = nn.Dropout(0.3)

    def forward(self,x):

        # x (B,T,312)

        x = x.permute(0,2,1)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)

        x = x.permute(0,2,1)

        x = self.attn(x)

        x = self.dropout(x)

        x = torch.relu(self.fc1(x))

        x = self.fc2(x)

        return x.squeeze()


# =====================================================
# 6 训练
# =====================================================

def train_model(model, train_loader, val_loader, device, epochs=30):

    pos_weight = torch.tensor([3.0]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    model.to(device)

    best_acc = 0

    for epoch in range(epochs):

        # ------------------
        # train
        # ------------------

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

            preds = (probs > 0.5).float()

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(train_labels,train_preds)

        # ------------------
        # validation
        # ------------------

        model.eval()

        val_preds = []
        val_labels = []

        with torch.no_grad():

            for X_batch,y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)

                probs = torch.sigmoid(outputs)

                preds = (probs > 0.5).float()

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

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(model.state_dict(),"slip_model_tcn.pth")


# =====================================================
# 7 主程序
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

    # =========================
    # 标准化
    # =========================

    mean = X_train.mean(axis=(0,1),keepdims=True)
    std = X_train.std(axis=(0,1),keepdims=True) + 1e-8

    X_train = (X_train-mean)/std
    X_val = (X_val-mean)/std

    print("train label:",np.unique(y_train,return_counts=True))

    # =========================
    # dataset
    # =========================

    train_dataset = TactileDataset(X_train,y_train,augment=True)
    val_dataset = TactileDataset(X_val,y_val)

    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:",device)

    model = SlipDetectionModel()

    train_model(model,train_loader,val_loader,device,epochs=20)

    print("训练完成")