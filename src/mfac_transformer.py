import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque

from tactile_sensor import TactileSensor
from so101_gripper import SO101ArmGripper


# =============================================================================
# 🔥 优化版 Transformer 模型定义 (D_MODEL=64 + Attention Pooling)
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
        
        # 🔴 Attention Pooling 层 (必须和训练时一致)
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
        x = self.proj(x)
        x = self.norm_proj(x)
        x = self.relu(x)
        
        x = x + self.pos_emb
        x = self.transformer(x)
        
        # 🔴 Attention Pooling (不再是 mean 了)
        attn_weights = self.attn(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x = torch.sum(x * attn_weights, dim=1)
        
        x = self.norm(x)
        x = self.drop(x)
        out = self.sigmoid(self.fc(x))
        return out.squeeze()

    def add_delta_feature(self, window):
        delta = np.diff(window, axis=0)
        delta = np.concatenate([np.zeros((1, window.shape[1])), delta], axis=0)
        window_new = np.concatenate([window, delta], axis=1)
        return window_new


# =============================================================================
# 🧠 滑动检测器（保持不变，兼容新模型）
# =============================================================================
class SlipDetector:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化默认参数
        self.model = None
        self.mean = 0.0
        self.std = 1.0
        
        # 滑动检测状态机参数
        self.high_th = 0.7
        self.low_th = 0.3
        self.state = 0
        self.slip_counter = 0
        
        try:
            print(f"正在加载模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.model = TinyTactileTransformer()
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("检测到完整 Checkpoint...")
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                if 'mean' in checkpoint and 'std' in checkpoint:
                    # 彻底的维度清洗
                    m = checkpoint['mean']
                    s = checkpoint['std']
                    if isinstance(m, torch.Tensor): m = m.cpu().numpy()
                    if isinstance(s, torch.Tensor): s = s.cpu().numpy()
                    self.mean = m.flatten().astype(np.float32)
                    self.std = s.flatten().astype(np.float32)
                    print(f"✅ 标准化参数加载成功: mean.shape={self.mean.shape}")
                else:
                    print("⚠️ 未找到 mean/std")
            else:
                print("检测到纯权重文件")
                self.model.load_state_dict(checkpoint)
                
        except Exception as e:
            print(f"❌ 模型加载出错: {e}")
            raise

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def predict(self, window):
        # 1. 加 Delta
        window = self.model.add_delta_feature(window)
        
        # 2. 标准化 (强制维度匹配)
        mean_reshaped = self.mean.reshape(1, -1)
        std_reshaped = self.std.reshape(1, -1)
        window = (window - mean_reshaped) / std_reshaped

        # 3. 转 Tensor 并加 Batch 维度
        window = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            prob = self.model(window).item()

        # 状态机逻辑
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
# 🚀 MFAC 控制器 (完全不变)
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
            print(" 第一个控制周期，仅初始化状态")
            self.is_first_control_cycle = False
            self._save_state(current_slip_prob, 0.0)
            return 0.0

        if self.y_prev is not None:
            delta_y = current_slip_prob - self.y_prev
            delta_u_prev = self.u_prev_step
            
            print(f" delta_y={delta_y:.6f}, delta_u_prev={delta_u_prev:.6f}, 当前phi={self.phi:.4f}")

            if abs(delta_u_prev) > 1e-8:
                phi_hat = self.phi + (self.rho * delta_u_prev / (self.mu + delta_u_prev ** 2)) * (delta_y - self.phi * delta_u_prev)
                self.phi = min(phi_hat, self.phi_limit)
            else:
                print(f"[PPD调试] 上一周期步长为0，跳过phi更新")

        error = current_slip_prob - self.s_ref
        if error <= 1e-6:
            print(f"[调试] 无滑动风险（误差={error:.6f}），不调整步长")
            self._save_state(current_slip_prob, 0.0)
            self.phi =  self.phi_init
            return 0.0

        delta_u_k = (self.eta * self.phi / (self.lambda_weight + abs(self.phi) ** 2)) * error
        step_increment = max(-delta_u_k, 0.0)
        step_increment = min(step_increment, self.max_single_step)

        self._save_state(current_slip_prob, step_increment)
        print("------step---------", step_increment)

        return step_increment

    def _save_state(self, current_y, current_step):
        self.y_prev = current_y
        self.u_prev_step = current_step
        self.u_total += current_step


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
    # 加载模型 
    # ======================
    try:
        #  修改路径：指向优化版训练生成的文件
        detector = SlipDetector("/home/liuli/tactile_lstm/models/transformer2_full.pth")
    except Exception as e:
        print("无法初始化检测器，程序退出")
        return

    # ======================
    # MFAC 控制器
    # ======================
    mfac_controller = SlipDrivenMFAC(
        s_ref=0.1,
        max_single_step=2,
        max_force=50.0
    )

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
            max_force = np.max(latest_frame)
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
                    else:
                        print("夹取力不平衡 → 打开夹爪终止抓取")
                        gripper.gripper_open()
                        return

            elif state == "GRASPING":
                if not gripper.force_balance(fa, fb):
                    print("警告：夹取力出现不平衡")
                
                if sensor.read_counts < 30:
                    continue
                
                if len(sensor.tactile_data_fifo) == 20:
                    window = np.array(sensor.tactile_data_fifo, dtype=np.float32)
                    start_time = time.time()
                    prob, confirmed_slip = detector.predict(window)
                    infer_time = (time.time() - start_time) * 1000

                    step_size = mfac_controller.update(current_slip_prob=prob)

                    print(f"滑动概率: {prob:.3f} | 推理耗时: {infer_time:.2f}ms | "
                          f"伪偏导数phi: {mfac_controller.phi:.3f} | 累计闭合步长: {mfac_controller.u_total:.3f} | "
                          f"本次执行步长: {step_size:.4f}")

                    if step_size > 1e-6:
                        gripper.mfac_close(step_size)

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
