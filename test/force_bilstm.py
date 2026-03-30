import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)
import os
import matplotlib.pyplot as plt
import time

# ==================== 全局设置：使用Linux默认英文字体 ====================
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# 1 Δforce feature (沿用变体1，继续启用)
def add_delta_feature(X):
    # X (N,T,312)
    delta = np.diff(X, axis=1)
    delta = np.concatenate(
        [np.zeros((X.shape[0],1,X.shape[2])), delta],
        axis=1
    )
    X_new = np.concatenate([X, delta], axis=2)
    print("After adding Δforce, shape:", X_new.shape)
    return X_new

# 2 Load specified files (与基线、变体1完全一致)
def load_npz_list(folder_path, file_list):
    X_list = []
    y_list = []
    for file in file_list:
        full_path = os.path.join(folder_path, file)
        print("Loading:", file)
        data = np.load(full_path)
        X = data["X"]
        y = data["y"]
        print("X shape:", X.shape)
        X_list.append(X)
        y_list.append(y)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    print("Combined shape:", X_all.shape)
    return X_all, y_all

# 3 Dataset (与基线、变体1完全一致)
class TactileDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 4 Model (变体2：仅将单向LSTM改为BiLSTM，其余与变体1一致)
class SlipDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 沿用变体1：输入维度624（原始312+Δforce312）
        self.feature_proj = nn.Linear(624, 128)
        self.relu = nn.ReLU()
        
        # ========== 变体2核心修改：单向LSTM → BiLSTM（bidirectional=True） ==========
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True  # 改为双向，仅此处修改
        )
        
        # 保持与变体1、基线一致：去掉Attention层，用全局平均池化
        self.dropout = nn.Dropout(0.3)
        
        # ========== 适配BiLSTM输出：双向LSTM输出维度=hidden_size×2=128，调整FC层输入 ==========
        self.fc1 = nn.Linear(128, 64)  # 从64改为128，适配双向输出
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x (B,T,624) - 沿用变体1的输入维度
        x = self.feature_proj(x)
        x = self.relu(x)
        
        # lstm_out shape: (B, T, 128) - 双向LSTM输出（64×2），与单向（64）区分
        lstm_out, _ = self.lstm(x)
        
        # 保持与变体1、基线一致：全局平均池化（无Attention）
        context = torch.mean(lstm_out, dim=1)
        
        out = self.dropout(context)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze()

# ==================== 5 Training (与基线、变体1完全一致，不做任何修改) ====================
def train_model(model, train_loader, val_loader, device, save_path, epochs=30, patience=5):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val_f1 = 0
    counter = 0
    model.to(device)
    for epoch in range(epochs):
        # ====================
        # Train
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
            preds = (outputs > 0.25).float()  # 与基线、变体1保持相同阈值
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())
        train_acc = accuracy_score(train_labels, train_preds)

        # ====================
        # Validation
        # ====================
        model.eval()
        val_preds = []
        val_labels = []
        val_scores = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                preds = (outputs > 0.25).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
                val_scores.extend(outputs.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss {total_loss:.3f} "
            f"TrainAcc {train_acc:.3f} "
            f"ValAcc {val_acc:.3f} "
            f"Precision {val_precision:.3f} "
            f"Recall {val_recall:.3f} "
            f"F1 {val_f1:.3f}"
        )

        # ====================
        # Early stopping
        # ====================
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            counter = 0
            print(f"Best model saved (Best Val F1: {best_val_f1:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break
    return best_val_f1

# ==================== 6 Evaluation (English Version，仅修改标题/文件名区分变体2) ====================
def plot_confusion_matrix(y_true, y_pred, classes, save_path, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """Plot and save confusion matrix (English version, no font issues)"""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, without normalization')
    print(cm)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=[0, 1],
           yticks=[0, 1],
           xticklabels=classes, 
           yticklabels=classes,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion Matrix saved to: {save_path}")
    plt.close()

def evaluate_full_model(model, val_loader, device, mean, std, model_save_path, output_dir):
    """Full evaluation pipeline (English)"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading best model from: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()

    y_true_all = []
    y_pred_all = []
    y_score_all = []
    inference_times = []
    
    print("Starting full validation inference...")
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            start_time = time.time()
            outputs = model(X_batch)
            end_time = time.time()
            batch_time = (end_time - start_time) * 1000 / len(X_batch)
            inference_times.append(batch_time)
            
            preds = (outputs > 0.25).float()
            y_true_all.extend(y_batch.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())
            y_score_all.extend(outputs.cpu().numpy())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_score_all = np.array(y_score_all)
    
    y_true_all = y_true_all.astype(int)
    y_pred_all = y_pred_all.astype(int)

    print("\n" + "="*50)
    print("Variant 2 (w/ Δforce + BiLSTM) Full Evaluation Results")  # 标注变体2
    print("="*50)
    acc = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    ap = average_precision_score(y_true_all, y_score_all)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AP Score:  {ap:.4f}")

    avg_inf_time = np.mean(inference_times)
    fps = 1000 / avg_inf_time
    print("\nReal-time Performance:")
    print(f"Avg Inference Time per Sample: {avg_inf_time:.2f} ms")
    print(f"Inference FPS:               {fps:.1f} FPS")

    class_names = np.array(['Stable', 'Slip'])
    # 文件名标注变体2，避免与其他模型混淆
    cm_path = os.path.join(output_dir, 'confusion_matrix_variant2.png')
    plot_confusion_matrix(y_true_all, y_pred_all, classes=class_names, save_path=cm_path, title='Variant 2 (w/ Δforce + BiLSTM) Confusion Matrix')

    precision_curve, recall_curve, _ = precision_recall_curve(y_true_all, y_score_all)
    # 文件名标注变体2
    pr_path = os.path.join(output_dir, 'pr_curve_variant2.png')
    
    plt.figure(figsize=(8, 6))
    plt.step(recall_curve, precision_curve, where='post', color='b', linewidth=1.2, label=f'PR Curve (AP={ap:.4f})')
    plt.fill_between(recall_curve, precision_curve, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Variant 2 (w/ Δforce + BiLSTM) Precision-Recall Curve', fontsize=14)  # 标注变体2
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    print(f"PR Curve saved to: {pr_path}")
    plt.close()

    return {
        'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'ap': ap,
        'avg_inf_time': avg_inf_time, 'fps': fps
    }

# 7 Main (核心修改：LSTM改为BiLSTM，修改保存路径/名称区分变体2)
if __name__ == "__main__":
    # ==================== Path Config (修改名称，明确区分变体2) ====================
    data_folder = "/home/liuli/tactile_lstm/train_data/data306"
    model_save_dir = "/home/liuli/tactile_lstm/models"
    eval_output_dir = "/home/liuli/tactile_lstm/eval_results/eval_results_variant2"  # 变体2单独文件夹
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_name = "lstm_variant2_bilstm"  # 标注变体2+BiLSTM
    model_save_path = os.path.join(model_save_dir, f"{model_name}.pth")
    final_save_path = os.path.join(model_save_dir, f"{model_name}_full.pth")

    # ==================== Data Loading (与变体1完全一致) ====================
    all_files = os.listdir(data_folder)
    train_files = [f for f in all_files if f.endswith(".npz") and not f.startswith("val_")]
    val_files = [f for f in all_files if f.endswith(".npz") and f.startswith("val_")]
    print("Train files:", train_files)
    print("Val files:", val_files)

    X_train, y_train = load_npz_list(data_folder, train_files)
    X_val, y_val = load_npz_list(data_folder, val_files)

    # 沿用变体1：启用Δforce特征添加（不修改）
    X_train = add_delta_feature(X_train)
    X_val = add_delta_feature(X_val)
    print("Variant 2: Using original 312-dim + Δforce 312-dim features + BiLSTM (No Attention)")

    # ==================== Normalization (与变体1完全一致) ====================
    mean = X_train.mean(axis=(0,1), keepdims=True)
    std = X_train.std(axis=(0,1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    print("Train label distribution:", np.unique(y_train, return_counts=True))

    # ==================== Dataset & DataLoader (与基线、变体1一致) ====================
    train_dataset = TactileDataset(X_train, y_train)
    val_dataset = TactileDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ==================== Device & Model ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = SlipDetectionModel()

    # ==================== Training ====================
    print("\n" + "="*50)
    print("Starting Variant 2 (w/ Δforce + BiLSTM) Training")  # 标注变体2
    print("="*50)
    best_f1 = train_model(model, train_loader, val_loader, device, model_save_path, epochs=30, patience=5)

    # ==================== Save Full Model ====================
    print(f"\nSaving full Variant 2 model (with norm params) to: {final_save_path}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "best_val_f1": best_f1
    }, final_save_path)

    # ==================== Full Evaluation ====================
    print("\n" + "="*50)
    print("Starting Variant 2 (w/ Δforce + BiLSTM) Model Evaluation")  # 标注变体2
    print("="*50)
    evaluate_full_model(
        model=model,
        val_loader=val_loader,
        device=device,
        mean=mean,
        std=std,
        model_save_path=model_save_path,
        output_dir=eval_output_dir
    )

    print("\nAll Variant 2 (w/ Δforce + BiLSTM) processes completed!")
