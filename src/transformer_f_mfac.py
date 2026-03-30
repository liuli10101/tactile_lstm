import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from tactile_sensor import TactileSensor
from so101_gripper import SO101ArmGripper

# =============================================================================
# ✅ 模型：625维，完全匹配你的 ftransformer.pth
# =============================================================================
class TinyTactileTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        INPUT_DIM = 625
        SEQ_LEN = 20
        D_MODEL = 32
        N_HEAD = 2
        NUM_LAYERS = 1
        
        self.proj = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_MODEL*2,
            batch_first=True, dropout=0.4, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(D_MODEL, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.proj(x) + self.pos_emb
        x = self.transformer(x)
        x = self.mean_pool(x)
        x = self.norm(x)
        x = self.drop(x)
        return self.sigmoid(self.fc(x)).squeeze()

    def mean_pool(self, x):
        return x.mean(dim=1)

    # 计算夹紧力
    def compute_force(self, frame):
        indices = np.arange(2, 312, 3)
        return np.sum(frame[indices]) / 2.0

    # 生成 625维 特征（触觉624 + 力1）
    def make_feature(self, window):
        # window: (20,312)
        delta = np.diff(window, axis=0)
        delta = np.concatenate([np.zeros((1,312)), delta], axis=0)
        feat624 = np.concatenate([window, delta], axis=1)
        
        # 加力维度 → 变成625
        force = np.array([[self.compute_force(f)] for f in window], dtype=np.float32)
        feat625 = np.concatenate([feat624, force], axis=1)
        return feat625, np.mean(force)


class SlipDrivenMFAC:
    def __init__(self,
                 s_ref=0,                
                 max_single_step=1,
                 max_force=40.0,
                 eta=3,
                 mu=0.01,
                 rho=0.95,
                 lambda_weight=0.1,
                 phi_init=-5.0
                 ):
        self.s_ref = s_ref
        self.max_single_step = max_single_step
        self.max_force = max_force
        
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
                if phi_hat >= -1e-6:
                    phi_hat = self.phi
                else:
                    self.phi = phi_hat

        error = current_slip_prob - self.s_ref
        if error <= 1e-6:
            self._save_state(current_slip_prob, 0.0)
            return 0.0

        delta_u_k = (self.eta * self.phi / (self.lambda_weight + abs(self.phi) ** 2)) * error
        step_increment = max(-delta_u_k, 0.0)
        step_increment = min(step_increment, self.max_single_step)

        self._save_state(current_slip_prob, step_increment)
        # return step_increment
        return self.u_total

    def _save_state(self, current_y, current_step):
        self.y_prev = current_y
        self.u_prev_step = current_step
        self.u_total += current_step
# =============================================================================
# ✅ 检测器
# =============================================================================
class SlipDetector:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = TinyTactileTransformer()
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, window):
        feat, current_F = self.model.make_feature(window)
        x = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(x).item(), current_F

# =============================================================================
# ✅ 主程序：滑动 → +1N → 直到不滑
# =============================================================================
def main():
    sensor = TactileSensor()
    gripper = SO101ArmGripper()
    sensor.tactile_data_fifo = deque(maxlen=20)

    if not sensor.connect() or not gripper.connect():
        return

    sensor.start_cycle_read()
    detector = SlipDetector("/home/liuli/tactile_lstm/models/f_transformer.pth")

    mfac_controller = SlipDrivenMFAC(
    max_single_step=2,
    max_force=50.0
    )

    SLIP_THRESH = 0.5
    ADD_FORCE = 1.0
    state = "WAIT"

    gripper.wrist_roll()
    time.sleep(0.05)
    gripper.gripper_open()
    gripper.gripper_close()

    while True:
        if len(sensor.tactile_data_fifo) < 20:
            time.sleep(0.02)
            continue

        win = np.array(sensor.tactile_data_fifo, dtype=np.float32)
        slip_prob, current_F = detector.predict(win)
        latest = sensor.tactile_data_fifo[-1]
        fa = [np.sum(latest[:156][::3]), np.sum(latest[:156][1::3]), np.sum(latest[:156][2::3])]
        fb = [np.sum(latest[156:][::3]), np.sum(latest[156:][1::3]), np.sum(latest[156:][2::3])]

        if state == "WAIT":
            if np.max(fa) > 0 and np.max(fb) > 0:
                gripper.gripper_stop()
                if gripper.force_balance(fa, fb):
                    state = "CONTROL"
                    mfac_controller.reset()
                    print("✅ 开始滑动控制")
                else:
                    gripper.gripper_open()
                    return

        elif state == "CONTROL":
            if slip_prob > SLIP_THRESH:
                

                step_size = mfac_controller.update()
                print(f"⚠️ 滑动！prob={slip_prob:.2f} | 当前力={current_F:.2f} → +1N")
                gripper.mfac_close(step_size)
            else:
                print(f"✅ 稳定 | prob={slip_prob:.2f} | 力={current_F:.2f}")

        time.sleep(0.05)

        try:
            pass
        except KeyboardInterrupt:
            break

    sensor.cycle_read_r = False
    gripper.gripper_stop()
    gripper.disconnect()
    sensor.disconnect()

if __name__ == "__main__":
    main()
