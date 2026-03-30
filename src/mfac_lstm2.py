import torch
import torch.nn as nn
import numpy as np
import time
import threading
from collections import deque

from tactile_sensor import TactileSensor
from so101_gripper import SO101ArmGripper


# 1 模型结构（完全保持不变）
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


# 2 滑移检测器（完全保持不变）
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


# 3 【核心修改】完全重写MFAC控制器，修复phi不变的致命bug
class SlipDrivenMFAC:
    def __init__(self,
                 s_ref=0.1,                # 期望滑动安全阈值
                 max_single_step=2,      # 【修改】降低单次最大步长，更安全
                #  max_total_step=5.0,       # 【修改】重新启用累计步长上限，防止无限累加
                 max_force=40.0,           # 最大法向力上限
                 # MFAC核心参数
                 eta=3,                   # 【修改】稍微增大步长因子，加快响应
                 mu=0.01,                   # 正则化因子
                 rho=0.95,                  # 遗忘因子
                 lambda_weight=0.1,         # 控制增量权重
                 phi_init=-5.0,              # 【修改】增大初始phi绝对值，初始控制更合理
                 phi_limit = -0.5
                 ):
        # 控制目标与安全约束
        self.s_ref = s_ref
        self.max_single_step = max_single_step
        # self.max_total_step = max_total_step
        self.max_force = max_force
        self.phi_limit = phi_limit
        # MFAC核心参数
        self.eta = eta
        self.mu = mu
        self.rho = rho
        self.lambda_weight = lambda_weight
        self.phi_init = phi_init

        # ======================
        # 【核心修改】正确的状态变量定义
        # ======================
        self.phi = phi_init               # 伪偏导数PPD
        self.y_prev = None                # 上一时刻的滑动概率 y(k-1)
        self.u_prev_step = 0.0            # 上一时刻执行的步长 Δu(k-1)
        self.u_total = 0.0                # 累计闭合总步长（仅用于安全约束）
        self.is_first_control_cycle = True # 标记是否为第一个有效控制周期

    def reset(self):
        """每次新抓取前必须调用，重置所有状态"""
        self.phi = self.phi_init
        self.y_prev = None
        self.u_prev_step = 0.0
        self.u_total = 0.0
        self.is_first_control_cycle = True

    def update(self, current_slip_prob):
        """
        MFAC控制周期更新
        """

        # 第一个有效周期：仅保存状态，不执行控制和PPD更新

        if self.is_first_control_cycle:
            print(" 第一个控制周期，仅初始化状态")
            self.is_first_control_cycle = False
            self._save_state(current_slip_prob, 0.0)
            return 0.0


        # PPD在线更新（严格遵循MFAC理论时序）

        if self.y_prev is not None:
            delta_y = current_slip_prob - self.y_prev  # Δy(k) = y(k) - y(k-1)
            delta_u_prev = self.u_prev_step            # Δu(k-1) = 上一次执行的步长

            # 【调试打印】关键变量，确认更新条件
            print(f" delta_y={delta_y:.6f}, delta_u_prev={delta_u_prev:.6f}, 当前phi={self.phi:.4f}")

            # 仅当输入有变化时更新PPD，避免分母为0
            if abs(delta_u_prev) > 1e-8:
                # 带遗忘因子的投影算法更新PPD
                phi_hat = self.phi + (self.rho * delta_u_prev / (self.mu + delta_u_prev ** 2)) * (delta_y - self.phi * delta_u_prev)
                self.phi = min(phi_hat,self.phi_limit)
                # 关键约束：phi必须为负（保证控制方向正确）
                # if phi_hat >= -1e-6:
                #     print(f"[PPD调试]phi符号错误，保留原值{self.phi:.4f}")
                #     phi_hat = self.phi
                # else:
                #     print(f"[PPD调试]phi更新成功：{self.phi:.4f} → {phi_hat:.4f}")
                #     self.phi = phi_hat
            else:
                print(f"[PPD调试] 上一周期步长为0，跳过phi更新")


        # 4. 计算滑动误差，无风险则直接返回

        error = current_slip_prob - self.s_ref
        if error <= 1e-6:
            print(f"[调试] 无滑动风险（误差={error:.6f}），不调整步长")
            self._save_state(current_slip_prob, 0.0)
            self.phi =  self.phi_init        #phi初始值
            return 0.0

        # ======================
        # 5. MFAC控制律计算（单向约束：只增不减）
        # ======================
        delta_u_k = (self.eta * self.phi / (self.lambda_weight + abs(self.phi) ** 2)) * error
        step_increment = max(-delta_u_k, 0.0)  # phi为负、error为正，取反得到正的闭合步长

        # 安全限幅
        step_increment = min(step_increment, self.max_single_step)
        # step_increment = min(step_increment, self.max_total_step - self.u_total)

        # ======================
        # 6. 统一保存状态
        # ======================
        self._save_state(current_slip_prob, step_increment)
        print("------step---------",step_increment )

        return step_increment

    def _save_state(self, current_y, current_step):
        """统一的状态保存函数，避免时序混乱"""
        self.y_prev = current_y          # 保存本次的滑动概率，作为下一周期的y(k-1)
        self.u_prev_step = current_step  # 保存本次执行的步长，作为下一周期的Δu(k-1)
        self.u_total += current_step     # 更新累计步长


# 4 主程序（仅适配MFAC初始化，其他完全保持你的原样）
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

    # FIFO数据窗口
    sensor.tactile_data_fifo = deque(maxlen=20)

    # 启动传感器采集线程
    if sensor.start_cycle_read():
        print("启动采集线程成功")

    # 初始化滑动检测器
    detector = SlipDetector("/home/liuli/tactile_lstm/models/lstm2_all.pth")
    
    # ======================
    # 初始化修复后的MFAC控制器
    # ======================
    mfac_controller = SlipDrivenMFAC(
        s_ref=0.1,                # 可根据实际效果微调
        max_single_step=2,      # 单次最大闭合步长
        # max_total_step=6.0,       # 累计最大闭合步长
        max_force=50.0            # 法向力安全上限
    )

    # 夹爪初始化
    gripper.wrist_roll()
    time.sleep(0.05)
    gripper.gripper_open()
    print("开始闭合夹爪，寻找接触点")
    gripper.gripper_close()

    # 抓取状态机
    state = "WAIT_CONTACT"

    try:
        while True:
            if len(sensor.tactile_data_fifo) == 0:
                time.sleep(0.05)
                continue

            # 读取最新传感器数据
            latest_frame = sensor.tactile_data_fifo[-1]
            max_force = np.max(latest_frame)
            a_force = latest_frame[:156]
            b_force = latest_frame[156:]
            fa = [np.sum(a_force[::3]), np.sum(a_force[1::3]), np.sum(a_force[2::3])]
            fb = [np.sum(b_force[::3]), np.sum(b_force[1::3]), np.sum(b_force[2::3])]

            # =============================
            # 状态机逻辑
            # =============================
            if state == "WAIT_CONTACT":
                if np.max(fa) > 0 and np.max(fb) > 0:
                    print("检测到物体接触 → 停止夹爪闭合")
                    gripper.gripper_stop()
                    if gripper.force_balance(fa, fb):
                        print("夹取力平衡正常 → 进入GRASPING状态")
                        state = "GRASPING"
                        # 【关键】进入抓取状态时，重置MFAC控制器
                        mfac_controller.reset()
                    else:
                        print("夹取力不平衡 → 打开夹爪终止抓取")
                        gripper.gripper_open()
                        return

            elif state == "GRASPING":
                if not gripper.force_balance(fa, fb):
                    print("警告：夹取力出现不平衡")
                
                if sensor.read_counts < 30:
                    print(f"等待数据窗口构建，当前采集次数={sensor.read_counts}")
                    continue
                
                if len(sensor.tactile_data_fifo) == 20:
                    window = np.array(sensor.tactile_data_fifo, dtype=np.float32)
                    start_time = time.time()
                    prob, confirmed_slip = detector.predict(window)
                    infer_time = (time.time() - start_time) * 1000

                    # ======================
                    # 核心：MFAC控制周期执行
                    # ======================
                    step_size = mfac_controller.update(
                        current_slip_prob=prob,
                    )

                    # 调试信息打印
                    print(f"滑动概率: {prob:.3f} | 推理耗时: {infer_time:.2f}ms | "
                          f"伪偏导数phi: {mfac_controller.phi:.3f} | 累计闭合步长: {mfac_controller.u_total:.3f} | "
                          f"本次执行步长: {step_size:.4f}")

                    # ======================
                    # 执行MFAC输出的步长指令
                    # ======================
                    if step_size > 1e-6:
                        print(f"执行MFAC自适应闭合：步长={step_size:.4f}")
                        gripper.mfac_close(step_size)

                    # ======================
                    # 保留紧急滑移处理（双重保险）
                    # ======================
                    # if confirmed_slip:
                    #     print("检测到持续滑移 → 紧急增力（双重保险）")
                    #     gripper.gripper_close()
                    #     time.sleep(0.05)
                    #     gripper.gripper_stop()

            # 控制周期延时
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("用户手动终止程序")

    finally:
        print("开始释放硬件资源")
        sensor.cycle_read_running = False
        time.sleep(0.5)
        gripper.gripper_stop()
        gripper.disconnect()
        sensor.disconnect()
        print("程序安全退出")


if __name__ == "__main__":
    main()
