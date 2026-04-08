import torch
import torch.nn as nn
import numpy as np
import time
import threading
import csv
import os
from collections import deque
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from tactile_sensor import TactileSensor
from so101_gripper import SO101ArmGripper


# =============================================================================
# 🔥 优化版 Transformer 模型 (D_MODEL=64 + 时间衰减 Attention)
# =============================================================================
class TinyTactileTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        INPUT_DIM = 624
        SEQ_LEN = 20
        D_MODEL = 64          # 🔴 必须和训练时一致：64
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
        # 🔥 保留你的强制修正逻辑作为双重保险
        batch_size = 1
        seq_len = 20
        feature_dim = 624
        x = x.reshape(batch_size, seq_len, feature_dim)
        
        # 正常前向传播
        x = self.proj(x)
        x = self.norm_proj(x)
        x = self.relu(x)
        
        x = x + self.pos_emb
        x = self.transformer(x)
        
        # 🔴 核心修改：时间衰减 Attention
        attn_weights = self.attn(x)
        
        # 生成时间衰减因子：从 0.5 线性增加到 1.0
        time_decay = torch.linspace(0.5, 1.0, steps=x.size(1), device=x.device)
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

    def add_delta_feature(self, window):
        if window.ndim == 3:
            window = window.squeeze(0) 
        delta = np.diff(window, axis=0)
        delta = np.concatenate([np.zeros((1, window.shape[1])), delta], axis=0)
        window_new = np.concatenate([window, delta], axis=1)
        return window_new


# =============================================================================
# 🧠 Detector（保持你的逻辑，增加维度清洗）
# =============================================================================
class SlipDetector:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Transformer model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = TinyTactileTransformer()
        
        # 兼容加载逻辑
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.mean = checkpoint["mean"]
            self.std = checkpoint["std"]
            
            # 🔴 维度清洗：确保 mean/std 是一维 numpy 数组
            m = self.mean
            s = self.std
            if isinstance(m, torch.Tensor): m = m.cpu().numpy()
            if isinstance(s, torch.Tensor): s = s.cpu().numpy()
            self.mean = m.flatten().astype(np.float32)
            self.std = s.flatten().astype(np.float32)
            
            print("✅ 已加载模型权重及标准化参数")
        else:
            self.model.load_state_dict(checkpoint)
            self.mean = 0.0
            self.std = 1.0
            print("⚠️  仅加载了模型权重")

        self.model.to(self.device)
        self.model.eval()

        self.high_th = 0.7
        self.low_th = 0.3
        self.state = 0
        self.slip_counter = 0

    def predict(self, window):
        # window 形状是 (20, 312)
        window = self.model.add_delta_feature(window) # 变成 (20, 624)
        
        # 🔴 标准化：确保维度匹配
        mean_reshaped = self.mean.reshape(1, -1)
        std_reshaped = self.std.reshape(1, -1)
        window = (window - mean_reshaped) / std_reshaped

        # 转 Tensor
        window = torch.tensor(window, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            prob = self.model(window).item()

        if prob > self.high_th:
            self.state = 1
        elif prob < self.low_th:
            self.state = 0

        if self.state == 1:
            self.slip_counter += 1
        else:
            self.slip_counter = 0

        confirmed_slip = self.slip_counter >= 3
        return prob, confirmed_slip


# =============================================================================
# 🚀 MFAC 控制器（完全保留）
# =============================================================================
class SlipDrivenMFAC:
    def __init__(self,
                 s_ref=0.1,
                 max_single_step=2,
                 max_force=40.0,
                 eta=3,
                 mu=0.01,
                 rho=0.95,
                 lambda_weight=0.1,
                 phi_init=-5.0,
                 phi_limit = -0.5
                 ):
        self.s_ref = s_ref
        self.max_single_step = max_single_step
        self.max_force = max_force
        self.phi_limit = phi_limit
        self.eta = eta
        self.mu = mu
        self.rho = rho
        self.lambda_weight = lambda_weight
        self.phi_init = phi_init

        self.phi = phi_init
        self.y_prev = None
        self.u_prev_step = 0.0
        self.u_total = 0.0
        self.is_first_control_cycle = True

    def reset(self):
        self.phi = self.phi_init
        self.y_prev = None
        self.u_prev_step = 0.0
        self.u_total = 0.0
        self.is_first_control_cycle = True

    def update(self, current_slip_prob):
        if self.is_first_control_cycle:
            self.is_first_control_cycle = False
            self._save_state(current_slip_prob, 0.0)
            return 0.0

        if self.y_prev is not None:
            delta_y = current_slip_prob - self.y_prev
            delta_u_prev = self.u_prev_step

            if abs(delta_u_prev) > 1e-8:
                phi_hat = self.phi + (self.rho * delta_u_prev / (self.mu + delta_u_prev ** 2)) * (delta_y - self.phi * delta_u_prev)
                self.phi = min(phi_hat, self.phi_limit)

        error = current_slip_prob - self.s_ref
        if error <= 1e-6:
            self._save_state(current_slip_prob, 0.0)
            self.phi = self.phi_init
            return 0.0

        delta_u_k = (self.eta * self.phi / (self.lambda_weight + abs(self.phi) ** 2)) * error
        step_increment = max(-delta_u_k, 0.0)
        step_increment = min(step_increment, self.max_single_step)

        self._save_state(current_slip_prob, step_increment)
        return step_increment

    def _save_state(self, current_y, current_step):
        self.y_prev = current_y
        self.u_prev_step = current_step
        self.u_total += current_step


# ==========================================
# 绘图函数：双Y轴合并图
# ==========================================
def plot_combined_curve(time_points, prob_points, step_points, adjustment_time_ms, total_closure, curve_index, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左Y轴：滑动概率
    color1 = '#d62728'
    ax1.set_xlabel('Time Since Slip Onset (s)', fontsize=14)
    ax1.set_ylabel('Slip Probability', color=color1, fontsize=14)
    line1, = ax1.plot(time_points, prob_points, color=color1, linewidth=2.5, label='Slip Probability')
    ax1.axhline(y=0.1, color=color1, linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (0.1)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)

    # 右Y轴：夹爪闭合量
    ax2 = ax1.twinx()
    color2 = '#2ca02c'
    ax2.set_ylabel('Cumulative Gripper Closure', color=color2, fontsize=14)
    line2, = ax2.plot(time_points, step_points, color=color2, linewidth=2.5, linestyle='-', label='Gripper Closure')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', fontsize=12)

    plt.title(f'Transformer Combined Response (Test #{curve_index})\nTime: {adjustment_time_ms:.2f} ms | Closure: {total_closure:.2f}', fontsize=16)
    plt.tight_layout()
    
    save_path = f"{save_folder}/combined_curve_{curve_index:03d}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# =============================================================================
# 🌍 主程序
# =============================================================================
def main():
    sensor = TactileSensor()
    gripper = SO101ArmGripper()

    print("连接设备...")
    if not sensor.connect():
        print("传感器连接失败")
        return
    if not gripper.connect():
        print("夹爪连接失败")
        return

    sensor.tactile_data_fifo = deque(maxlen=20)
    if sensor.start_cycle_read():
        print("启动采集线程成功")

    # ======================
    # 配置区域 (注意：这里要加载新训练的模型)
    # ======================
    TRANSFORMER_MODEL_PATH = "/home/liuli/tactile_lstm/models/transformer2_full.pth" # 🔴 路径已更新
    BASE_SAVE_PATH = "/home/liuli/tactile_lstm/experiment/transformer_gripper_result"
    SLIP_THRESHOLD = 0.1
    MAX_RECORD_TIME = 10.0

    # 初始化
    detector = SlipDetector(TRANSFORMER_MODEL_PATH)
    mfac_controller = SlipDrivenMFAC(
        s_ref=0.1, max_single_step=2, max_force=50.0
    )

    # 结果保存路径
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join(BASE_SAVE_PATH, f"transformer_exp_{timestamp_str}")
    os.makedirs(result_folder, exist_ok=True)
    summary_csv_path = os.path.join(result_folder, "experiment_summary.csv")
    raw_data_csv_path = os.path.join(result_folder, "raw_timeseries_data.csv")
    
    print(f"\n📁 [Transformer Decay] 实验结果将保存至: {result_folder}")
    print(f"🔄  每次实验开始时，闭合量计数器将自动清零\n")

    # 状态变量
    is_recording = False
    slip_start_time = 0.0
    curve_counter = 0
    adjustment_summary_records = []
    current_raw_data = []

    # 硬件初始化
    gripper.wrist_roll()
    time.sleep(0.05)
    gripper.gripper_open()
    print("开始闭合夹爪，寻找接触点")
    gripper.gripper_close()
    state = "WAIT_CONTACT"

    try:
        while True:
            if len(sensor.tactile_data_fifo) == 0:
                time.sleep(0.05)
                continue

            latest_frame = sensor.tactile_data_fifo[-1]
            current_total_force = np.sum(latest_frame) / 2
            a_force = latest_frame[:156]
            b_force = latest_frame[156:]
            fa = [np.sum(a_force[::3]), np.sum(a_force[1::3]), np.sum(a_force[2::3])]
            fb = [np.sum(b_force[::3]), np.sum(b_force[1::3]), np.sum(b_force[2::3])]

            if state == "WAIT_CONTACT":
                if np.max(fa) > 0 and np.max(fb) > 0:
                    print("检测到物体接触 → 停止夹爪闭合")
                    gripper.gripper_stop()
                    if gripper.force_balance(fa, fb):
                        print("夹取力平衡正常 → 进入GRASPING状态")
                        state = "GRASPING"
                        mfac_controller.reset()
                        is_recording = False
                        slip_start_time = 0.0
                        curve_counter = 0
                        current_raw_data = []
                        print("✅ 实验状态已重置")
                    else:
                        print("夹取力不平衡 → 打开夹爪终止抓取")
                        gripper.gripper_open()
                        return

            elif state == "GRASPING":
                if not gripper.force_balance(fa, fb):
                    pass
                
                if sensor.read_counts < 30:
                    continue
                
                if len(sensor.tactile_data_fifo) == 20:
                    window = np.array(sensor.tactile_data_fifo, dtype=np.float32)
                    prob, confirmed_slip = detector.predict(window)
                    step_size = mfac_controller.update(current_slip_prob=prob)
                    current_time = time.time()

                    # ==========================================
                    # 🔥 核心记录逻辑 + 闭合量清零
                    # ==========================================
                    
                    # 1. 开始记录
                    if prob > SLIP_THRESHOLD and not is_recording:
                        slip_start_time = current_time
                        is_recording = True
                        current_raw_data = []
                        
                        # 🔥 清零闭合量
                        mfac_controller.u_total = 0.0
                        mfac_controller.u_prev_step = 0.0 
                        
                        print(f"\n⚠️  [开始记录] 滑动概率: {prob:.3f} > {SLIP_THRESHOLD}")
                        print(f"   🔄 闭合量计数器已清零")

                    # 2. 正在记录
                    if is_recording:
                        rel_time = current_time - slip_start_time
                        current_raw_data.append({
                            "abs_time": current_time,
                            "rel_time": rel_time,
                            "rel_time_ms": rel_time * 1000,
                            "slip_prob": prob,
                            "gripper_step": mfac_controller.u_total,
                            "delta_step": step_size,
                            "phi": mfac_controller.phi,
                            "total_force": current_total_force
                        })

                        # 3. 结束记录
                        if prob < SLIP_THRESHOLD or rel_time > MAX_RECORD_TIME:
                            total_adjustment_time = (current_time - slip_start_time) * 1000
                            total_closure = mfac_controller.u_total
                            curve_counter += 1
                            
                            if rel_time > MAX_RECORD_TIME:
                                print(f"\n⏰ [超时结束]")
                            else:
                                print(f"\n✅ [结束记录] 滑动概率: {prob:.3f} < {SLIP_THRESHOLD}")
                            
                            print(f"⏱️  总耗时: {total_adjustment_time:.2f} ms | 本次闭合量: {total_closure:.2f}")

                            # 保存摘要
                            adjustment_summary_records.append([
                                curve_counter, slip_start_time, current_time,
                                total_adjustment_time, total_closure
                            ])

                            # 生成曲线
                            times = [d["rel_time"] for d in current_raw_data]
                            probs = [d["slip_prob"] for d in current_raw_data]
                            steps = [d["gripper_step"] for d in current_raw_data]
                            
                            img_path = plot_combined_curve(
                                times, probs, steps, 
                                total_adjustment_time, total_closure,
                                curve_counter, result_folder
                            )
                            print(f"📈 已生成曲线: {img_path}")

                            # 保存原始CSV
                            file_exists = os.path.isfile(raw_data_csv_path)
                            with open(raw_data_csv_path, 'a', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=current_raw_data[0].keys())
                                if not file_exists:
                                    writer.writeheader()
                                for row in current_raw_data:
                                    row['exp_id'] = curve_counter
                                    writer.writerow(row)
                            
                            print(f"💾 数据已保存 (Exp #{curve_counter})\n")

                            # 重置
                            is_recording = False
                            slip_start_time = 0.0
                            current_raw_data = []

                    # 调试打印
                    status_str = "📝 RECORDING" if is_recording else "🟢 Monitoring"
                    print(f"\r[{status_str}] Force: {current_total_force:.1f} | Prob: {prob:.3f} | Step: {mfac_controller.u_total:.2f}", end="", flush=True)

                    if step_size > 1e-6:
                        gripper.mfac_close(step_size)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n用户手动终止程序")

    finally:
        print(f"\n💾 正在保存实验摘要...")
        if len(adjustment_summary_records) > 0:
            try:
                with open(summary_csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['exp_id', 'start_time', 'end_time', 'adjustment_time_ms', 'total_closure_units'])
                    for rec in adjustment_summary_records:
                        writer.writerow(rec)
                
                times_ms = [r[3] for r in adjustment_summary_records]
                closures = [r[4] for r in adjustment_summary_records]
                print(f"\n🎉 [Transformer Decay] 实验结束！")
                print(f"📂 结果文件夹: {result_folder}")
                print("="*70)
                print(f"📊 总实验次数: {len(times_ms)}")
                print(f"📊 平均调整时间: {np.mean(times_ms):.2f} ms")
                print(f"📊 平均闭合量:   {np.mean(closures):.2f} units")
                print("="*70)
            except Exception as e:
                print(f"❌ 保存失败: {e}")

        print("\n开始释放硬件资源")
        sensor.cycle_read_running = False
        time.sleep(0.5)
        gripper.gripper_stop()
        gripper.disconnect()
        sensor.disconnect()
        print("程序安全退出")


if __name__ == "__main__":
    main()
