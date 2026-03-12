import torch
import torch.nn as nn
import numpy as np
import time
import threading
from collections import deque

from tactile_sensor import TactileSensor
from so101_gripper import SO101ArmGripper


# ======================================
# 1️⃣ 与训练完全一致的模型结构
# ======================================
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

class AttentionPooling(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # x : (B,T,H)

        weights = torch.softmax(self.attn(x), dim=1)

        out = torch.sum(x * weights, dim=1)

        return out

# ======================================
# 2️⃣ 滑移检测器
# ======================================
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
        self.slip_counter = 0  # 连续滑移计数

    def predict(self, window):


        window = torch.tensor(window, dtype=torch.float32)

        # (20,312) -> (1,20,312)
        window = window.unsqueeze(0)

        window = window.to(self.device)
        with torch.no_grad():

            logit = self.model(window)

            prob = torch.sigmoid(logit).item()

        # 双阈值滞回
        if prob > self.high_th:
            self.state = 1
        elif prob < self.low_th:
            self.state = 0

        # 连续3次才触发
        if self.state == 1:
            self.slip_counter += 1
        else:
            self.slip_counter = 0

        confirmed_slip = self.slip_counter >= 3

        return prob, confirmed_slip


# ======================================
# 3️⃣ 主程序
# ======================================
def main():

    sensor = TactileSensor()
    gripper = SO101ArmGripper()
    # 确认设备初始化成功

    print("连接设备...")
    if not sensor.connect():
        print("传感器连接失败")
        return

    if not gripper.connect():
        print("夹爪连接失败")
        return

    # FIFO窗口
    sensor.tactile_data_fifo = deque(maxlen=20)


    # 启动采集线程
    if (sensor.start_cycle_read()):
        print("启动采集线程")


    detector = SlipDetector(
        "/home/liuli/tactile_lstm/models/slip_model_cba.pth"
    )
    
    gripper.gripper_open()  #打开夹爪
    print("闭合夹爪")
    gripper.gripper_close()

    state = "WAIT_CONTACT"

    try:
        while True:

            if len(sensor.tactile_data_fifo) == 0:
                time.sleep(0.05)
                continue

            # print(len(sensor.tactile_data_fifo))
            latest_frame = sensor.tactile_data_fifo[-1]

            print(latest_frame)

            max_force = np.max(latest_frame)
            a_force = latest_frame[:156]
            b_force = latest_frame[156:]
            fa = [np.sum(a_force[::3]),np.sum(a_force[1::3]),np.sum(a_force[2::3])] #拇指
            fb = [np.sum(b_force[::3]),np.sum(b_force[1::3]),np.sum(b_force[2::3])] #中指


            # =============================
            # 状态机
            # =============================

            if state == "WAIT_CONTACT":

                if max_force > 0:
                    print("检测到接触 → 停止闭合")
                    gripper.gripper_stop()
                    if gripper.force_balance(fa,fb):
                        print("夹取力方向在稳定范围内")
                        state = "GRASPING"
                    else:
                        print("夹取力不稳定")
                        gripper.gripper_open()  #打开夹爪
                        return
            

            elif state == "GRASPING":
                # !!!!!!!!!!!!!!!!!?
                if gripper.force_balance(fa,fb):
                    print("夹取力方向在稳定范围内")

                if sensor.read_counts < 30:
                    print("等待数据窗口构建完成,当前循环次数=",sensor.read_counts)
                    continue
                if len(sensor.tactile_data_fifo) == 20:

                    window = np.array(sensor.tactile_data_fifo, dtype=np.float32)
                    # print(sensor.tactile_data_fifo)
                    start_time = time.time()
                    prob, slip = detector.predict(window)
                    infer_time = (time.time() - start_time) * 1000

                    print(f"Slip prob: {prob:.3f} | 推理耗时: {infer_time:.2f} ms")

                    if slip:
                        print("确认滑移 → 增力")

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
