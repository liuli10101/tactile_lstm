import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, 
                             precision_recall_curve)
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格和字体（避免中文乱码）
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 如果需要中文，可改为 'SimHei' 或 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心函数（保持不变）
# ==========================================

def add_delta_feature(X):
    delta = np.diff(X, axis=1)
    delta = np.concatenate(
        [np.zeros((X.shape[0], 1, X.shape[2])), delta],
        axis=1
    )
    X_new = np.concatenate([X, delta], axis=2)
    print("加入Δforce后 shape:", X_new.shape)
    return X_new

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

class TactileDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SlipDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_proj = nn.Linear(624, 128)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.attn = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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

# ==========================================
# 2. 测试核心逻辑（返回延迟数据）
# ==========================================
def test_model(model, test_loader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_infer_times = []  # 🔥 保存每个样本的推理延迟

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            outputs = model(X_batch)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            batch_time = (time.time() - start_time) * 1000
            
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            
            batch_size = X_batch.shape[0]
            all_infer_times.extend([batch_time / batch_size] * batch_size)

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0

    mean_latency = np.mean(all_infer_times)
    std_latency = np.std(all_infer_times)
    max_latency = np.max(all_infer_times)
    p99_latency = np.percentile(all_infer_times, 99)

    # 打印结果
    print("\n" + "="*70)
    print("【测试结果汇总】")
    print("="*70)
    print(f"样本总数: {len(all_labels)}")
    print(f"\n[分类性能]")
    print(f"  准确率 (Accuracy):  {acc:.4f}")
    print(f"  精确率 (Precision): {prec:.4f}")
    print(f"  召回率 (Recall):    {rec:.4f}")
    print(f"  F1 值 (F1-Score):   {f1:.4f}")
    print(f"\n[混淆矩阵]")
    print(f"  TP (真正例): {tp} | FP (假正例): {fp}")
    print(f"  TN (真负例): {tn} | FN (假负例): {fn}")
    print(f"\n[推理延迟统计 (单样本)]")
    print(f"  平均延迟: {mean_latency:.2f} ms")
    print(f"  标准差:   {std_latency:.2f} ms")
    print(f"  最大延迟: {max_latency:.2f} ms")
    print(f"  99分位:   {p99_latency:.2f} ms")
    print("="*70)

    # 🔥 额外返回 all_infer_times 用于画图
    return (all_labels, all_preds, all_probs, all_infer_times,
            {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

# ==========================================
# 3. 可视化绘图函数（增加延迟图）
# ==========================================
def plot_visualizations(y_true, y_pred, y_prob, latencies, save_path="test_results"):
    os.makedirs(save_path, exist_ok=True)
    class_names = ["Stable (0)", "Slip (1)"]
    
    # 1. 混淆矩阵热力图
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 16})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300)
    print(f"✅ 混淆矩阵已保存")
    plt.close()

    # 2. ROC 曲线
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/roc_curve.png", dpi=300)
    print(f"✅ ROC曲线已保存")
    plt.close()

    # 3. PR 曲线
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, color='darkgreen', lw=2,
             label=f'PR curve (area = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/pr_curve.png", dpi=300)
    print(f"✅ PR曲线已保存")
    plt.close()

    # 4. 🔥 推理延迟分布直方图（和Transformer一模一样）
    mean_lat = np.mean(latencies)
    p99_lat = np.percentile(latencies, 99)
    
    plt.figure(figsize=(8, 6))
    sns.histplot(latencies, bins=15, kde=True, color='steelblue')
    plt.axvline(mean_lat, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_lat:.2f} ms')
    plt.axvline(p99_lat, color='orange', linestyle='--', linewidth=2, label=f'P99: {p99_lat:.2f} ms')
    
    plt.title('Inference Latency Distribution', fontsize=16)
    plt.xlabel('Latency (ms)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/latency_distribution.png", dpi=300)
    print(f"✅ 推理延迟统计图已保存")
    plt.close()

    print("\n🎉 所有可视化图表生成完毕！")

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # =========================
    # 【配置区域】
    # =========================
    data_folder = "/home/liuli/tactile_lstm/train_data/experiment" 
    model_path = "/home/liuli/tactile_lstm/models/lstm2_all.pth"
    output_img_folder = "cylinder_lstmtest_figs"
    
    test_files = [
        "cylinder_stable.npz",
        "cylinder_slip.npz"
    ]

    # =========================
    # 环境准备
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # =========================
    # 1. 加载数据
    # =========================
    print("\n--- 1. 加载测试数据 ---")
    X_test, y_test = load_npz_list(data_folder, test_files)

    # =========================
    # 2. 加载模型
    # =========================
    print("\n--- 2. 加载模型 ---")
    checkpoint = torch.load(model_path, map_location=device)
    model = SlipDetectionModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    # =========================
    # 3. 预处理
    # =========================
    print("\n--- 3. 数据预处理 ---")
    X_test = add_delta_feature(X_test)
    X_test = (X_test - mean) / std
    print("测试集标签分布:", np.unique(y_test, return_counts=True))

    # =========================
    # 4. DataLoader
    # =========================
    test_dataset = TactileDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # =========================
    # 5. 运行测试
    # =========================
    print("\n--- 4. 开始推理 ---")
    y_true, y_pred, y_prob, latencies, metrics = test_model(model, test_loader, device)

    # =========================
    # 6. 生成可视化图表（传入延迟）
    # =========================
    print("\n--- 5. 生成可视化图表 ---")
    plot_visualizations(y_true, y_pred, y_prob, latencies, save_path=output_img_folder)
