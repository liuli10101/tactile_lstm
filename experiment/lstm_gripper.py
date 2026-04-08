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
matplotlib.use('Agg') # 强制使用非交互式后端，防止报错
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from tactile_sensor import TactileSensor
from so101_gripper import SO101ArmGripper


# 1 模型结构
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
    
    def add_delta_feature(self, window):
        delta = np.diff(window, axis=0)
        delta = np.concatenate([np.zeros((1, window.shape[1])), delta], axis=0)
        window_new = np.concatenate([window, delta], axis=1)
        return window_new


# 2 滑移检测器
class SlipDetector:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = SlipDetectionModel()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        self.high_th = 0.7
        self.low_th = 0.3
        self.state = 0
        self.slip_counter = 0

    def predict(self, window):
        window = self.add_delta_feature(window)
        window = (window - self.mean) / self.std
        window = torch.tensor(window, dtype=torch.float32).to(self.device)
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
    
    def add_delta_feature(self, window):
        delta = np.diff(window, axis=0)
        delta = np.concatenate([np.zeros((1, window.shape[1])), delta], axis=0)
        window = np.concatenate([window, delta], axis=1)
        return window


# 3 MFAC控制器
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
        
    def reset_closure_only(self):
        """🔥 新增：仅重置闭合量计数器，保留phi等控制参数"""
        self.u_total = 0.0
        # 注意：这里不重置 phi 和 y_prev，保持控制器的连续性

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
    
    # 图例合并
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', fontsize=12)

    plt.title(f'Combined Response Curve (Test #{curve_index})\nAdjustment Time: {adjustment_time_ms:.2f} ms | Total Closure: {total_closure:.2f}', fontsize=16)
    plt.tight_layout()
    
    save_path = f"{save_folder}/combined_curve_{curve_index:03d}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# 4 主程序
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

    detector = SlipDetector("/home/liuli/tactile_lstm/models/lstm2_all.pth")
    mfac_controller = SlipDrivenMFAC(
        s_ref=0.1,
        max_force=50.0
    )

    # ==========================================
    # 实验配置
    # ==========================================
    SLIP_THRESHOLD = 0.1
    MAX_RECORD_TIME = 10.0
    BASE_SAVE_PATH = "/home/liuli/tactile_lstm/experiment/lstm_gripper_result"
    
    # 状态变量
    is_recording = False
    slip_start_time = 0.0
    curve_counter = 0
    
    # 数据记录容器
    adjustment_summary_records = []
    current_raw_data = []
    
    # 结果保存
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join(BASE_SAVE_PATH, f"lstm_exp_{timestamp_str}")
    os.makedirs(result_folder, exist_ok=True)
    
    summary_csv_path = os.path.join(result_folder, "experiment_summary.csv")
    raw_data_csv_path = os.path.join(result_folder, "raw_timeseries_data.csv")
    
    print(f"\n📁 实验结果将保存至: {result_folder}")
    print(f"⏱️  触发条件: Prob > {SLIP_THRESHOLD} (开始) / Prob < {SLIP_THRESHOLD} (结束)")
    print(f"🔄  每次实验开始时，闭合量计数器将自动清零\n")
    # ==========================================

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
                        
                        # 重置
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
                    # 🔥 核心修改逻辑
                    # ==========================================
                    
                    # 1. 开始记录：概率 > 0.1 且未在记录
                    if prob > SLIP_THRESHOLD and not is_recording:
                        slip_start_time = current_time
                        is_recording = True
                        current_raw_data = []
                        
                        # 🔥🔥🔥 关键修改：每次开始记录前，将闭合量清零
                        mfac_controller.u_total = 0.0
                        # 同时也重置一下上一步的步长，防止影响
                        mfac_controller.u_prev_step = 0.0 
                        
                        print(f"\n⚠️  [开始记录] 滑动概率: {prob:.3f} > {SLIP_THRESHOLD}")
                        print(f"   🔄 闭合量计数器已清零")

                    # 2. 正在记录：保存数据
                    if is_recording:
                        rel_time = current_time - slip_start_time
                        # 记录所有详细数据
                        current_raw_data.append({
                            "abs_time": current_time,
                            "rel_time": rel_time,
                            "rel_time_ms": rel_time * 1000,
                            "slip_prob": prob,
                            "gripper_step": mfac_controller.u_total, # 这里现在是从0开始的
                            "delta_step": step_size,
                            "phi": mfac_controller.phi,
                            "total_force": current_total_force
                        })

                        # 3. 结束记录：概率 < 0.1 或者 超时
                        if prob < SLIP_THRESHOLD or rel_time > MAX_RECORD_TIME:
                            total_adjustment_time = (current_time - slip_start_time) * 1000
                            total_closure = mfac_controller.u_total # 这里的数值现在也是从0开始算的
                            curve_counter += 1
                            
                            if rel_time > MAX_RECORD_TIME:
                                print(f"\n⏰ [超时结束] 记录已超过{MAX_RECORD_TIME}秒，自动终止")
                            else:
                                print(f"\n✅ [结束记录] 滑动概率: {prob:.3f} < {SLIP_THRESHOLD}")
                            
                            print(f"⏱️  总耗时: {total_adjustment_time:.2f} ms | 本次闭合量: {total_closure:.2f}")

                            # 保存摘要数据
                            adjustment_summary_records.append([
                                curve_counter,
                                slip_start_time,
                                current_time,
                                total_adjustment_time,
                                total_closure
                            ])

                            # 生成合并曲线
                            times = [d["rel_time"] for d in current_raw_data]
                            probs = [d["slip_prob"] for d in current_raw_data]
                            steps = [d["gripper_step"] for d in current_raw_data]
                            
                            img_path = plot_combined_curve(
                                times, probs, steps, 
                                total_adjustment_time, total_closure,
                                curve_counter, result_folder
                            )
                            print(f"📈 已生成合并曲线: {img_path}")

                            # 保存原始数据到CSV
                            file_exists = os.path.isfile(raw_data_csv_path)
                            with open(raw_data_csv_path, 'a', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=current_raw_data[0].keys())
                                if not file_exists:
                                    writer.writeheader()
                                for row in current_raw_data:
                                    row['exp_id'] = curve_counter
                                    writer.writerow(row)
                            
                            print(f"💾 原始数据已追加至 CSV (Exp #{curve_counter})\n")

                            # 重置状态
                            is_recording = False
                            slip_start_time = 0.0
                            current_raw_data = []
                            # 注意：这里不清零 u_total，等下一次开始记录时再清零

                    # 调试打印
                    status_str = "📝 RECORDING" if is_recording else "🟢 Monitoring"
                    print(f"\r[{status_str}] Force: {current_total_force:.1f} | Prob: {prob:.3f} | Step: {mfac_controller.u_total:.2f}", end="", flush=True)

                    if step_size > 1e-6:
                        gripper.mfac_close(step_size)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n用户手动终止程序")

    finally:
        # ==========================================
        # 最后保存摘要CSV
        # ==========================================
        print(f"\n💾 正在保存实验摘要...")
        if len(adjustment_summary_records) > 0:
            try:
                with open(summary_csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        'exp_id', 'start_time', 'end_time', 
                        'adjustment_time_ms', 'total_closure_units'
                    ])
                    for rec in adjustment_summary_records:
                        writer.writerow(rec)
                
                # 打印统计
                times_ms = [r[3] for r in adjustment_summary_records]
                closures = [r[4] for r in adjustment_summary_records]
                print(f"\n🎉 实验结束！")
                print(f"📂 结果文件夹: {result_folder}")
                print("="*70)
                print(f"📊 总实验次数: {len(times_ms)}")
                print(f"📊 平均调整时间: {np.mean(times_ms):.2f} ms")
                print(f"📊 平均单次闭合量: {np.mean(closures):.2f} units")
                print("="*70)
                
            except Exception as e:
                print(f"❌ 保存失败: {e}")
        else:
            print("未记录到有效数据。")

        # 释放资源
        print("\n开始释放硬件资源")
        sensor.cycle_read_running = False
        time.sleep(0.5)
        gripper.gripper_stop()
        gripper.disconnect()
        sensor.disconnect()
        print("程序安全退出")


if __name__ == "__main__":
    main()
