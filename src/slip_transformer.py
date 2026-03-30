import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque

from tactile_sensor import TactileSensor
from so101_gripper import SO101ArmGripper


# =============================================================================
# 🔥 替换成你自己的 TRANSFORMER 模型（完全对齐接口）
# =============================================================================
class TinyTactileTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        INPUT_DIM = 624
        SEQ_LEN = 20
        D_MODEL = 32
        N_HEAD = 2
        NUM_LAYERS = 1
        
        self.proj = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=D_MODEL * 2,
            batch_first=True,
            dropout=0.4,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        
        self.norm = nn.LayerNorm(D_MODEL)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(D_MODEL, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = (B, 20, 624)
        x = self.proj(x)
        x = x + self.pos_emb
        x = self.transformer(x)
        x = x.mean(dim=1)  # 20帧 → 1个输出
        x = self.norm(x)
        x = self.drop(x)
        out = self.sigmoid(self.fc(x))
        return out.squeeze()  # 输出和LSTM完全一样：标量概率


# =============================================================================
# 滑移检测器（模型替换，其余完全不变）
# =============================================================================
class SlipDetector:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 🔥 这里换成 Transformer
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = TinyTactileTransformer()
        self.model.load_state_dict(checkpoint)  # Transformer权重没有嵌套
        self.model.to(self.device)
        self.model.eval()

        # 你训练Transformer时有没有保存 mean / std？
        # 如果没有，用下面默认值（建议训练时保存）
        self.mean = 0.0
        self.std = 1.0

        self.high_th = 0.7
        self.low_th = 0.3
        self.state = 0
        self.slip_counter = 0

    def predict(self, window):
        window = self.add_delta_feature(window)  # (20,624)
        window = (window - self.mean) / self.std

        window = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,20,624)

        with torch.no_grad():
            prob = self.model(window).item()

        # 滞回逻辑不变
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


# =============================================================================
# 主程序完全不变！
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
        print("启动采集线程")

    # 🔥 加载你的 TRANSFORMER 模型
    detector = SlipDetector("/home/liuli/tactile_lstm/models/best_transformer.pth")

    gripper.wrist_roll()
    time.sleep(0.05)
    gripper.gripper_open()
    print("闭合夹爪")
    gripper.gripper_close()

    state = "WAIT_CONTACT"

    try:
        while True:
            if len(sensor.tactile_data_fifo) == 0:
                time.sleep(0.05)
                continue

            latest_frame = sensor.tactile_data_fifo[-1]
            a_force = latest_frame[:156]
            b_force = latest_frame[156:]
            fa = [np.sum(a_force[::3]), np.sum(a_force[1::3]), np.sum(a_force[2::3])]
            fb = [np.sum(b_force[::3]), np.sum(b_force[1::3]), np.sum(b_force[2::3])]

            if state == "WAIT_CONTACT":
                if  np.max(fa) > 0 and np.max(fb) > 0:
                    print("检测到接触 → 停止闭合")
                    gripper.gripper_stop()
                    if gripper.force_balance(fa, fb):
                        print("夹取力方向在稳定范围内")
                        state = "GRASPING"
                    else:
                        print("夹取力不稳定")
                        gripper.gripper_open()
                        return

            elif state == "GRASPING":
                if gripper.force_balance(fa, fb):
                    print("夹取力在稳定范围内")
                
                if sensor.read_counts < 30:
                    continue
                if len(sensor.tactile_data_fifo) == 20:
                    window = np.array(sensor.tactile_data_fifo, dtype=np.float32)
                    start_time = time.time()
                    prob, slip = detector.predict(window)
                    infer_time = (time.time() - start_time) * 1000

                    print(f"Slip prob: {prob:.3f} | 推理耗时: {infer_time:.2f} ms")

                    if slip:
                        print("确认滑移 → 增力")
                        gripper.wrist_roll()
                        gripper.gripper_close()
                        time.sleep(0.1)
                        gripper.gripper_stop()

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("用户终止")

    finally:
        print("释放资源")
        sensor.cycle_read_running = False
        time.sleep(0.5)
        gripper.gripper_stop()
        gripper.disconnect()
        sensor.disconnect()
        print("程序结束")


if __name__ == "__main__":
    main()
