import numpy as np
import time
from collections import deque
import pandas as pd

from tactile_sensor import TactileSensor
from so101_gripper import SO101ArmGripper

# ================= 数据记录器 =================
class DataRecorder:
    def __init__(self):
        self.start_time = None
        self.data = {
            'time': [],                # 相对时间戳(s)
            'input_alpha': [],         # 输入：过盈量相对指令(°)
            'output_force_left': [],   # 输出：左指总法向力
            'output_force_right': [],  # 输出：右指总法向力
            'output_force_total': []   # 输出：两指平均法向力
        }
    
    def start(self):
        self.start_time = time.time()
    
    def record(self, input_val, force_left, force_right):
        if self.start_time is None:
            return
        t = time.time() - self.start_time
        total_force = (force_left + force_right) / 2
        self.data['time'].append(t)
        self.data['input_alpha'].append(input_val)
        self.data['output_force_left'].append(force_left)
        self.data['output_force_right'].append(force_right)
        self.data['output_force_total'].append(total_force)
    
    def save_csv(self, filename="alpha_to_force_step_response2.csv"):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"[DataRecorder] 数据已保存至: {filename}")
        print(f"[DataRecorder] 共记录 {len(df)} 组数据")

def main():
    # ================= 实验参数配置区（按需修改） =================
    STEP_AMPLITUDE = 1  # 阶跃幅值：额外闭合的角度(°)，建议0.1~0.5°，不夹坏工件为准
    STEADY_WAIT_BEFORE_STEP = 0  # 接触后等待稳定的时间(s)
    RECORD_AFTER_STEP = 1.0         # 阶跃后继续记录的时间(s)
    CONTACT_FORCE_THRESHOLD = 0.1     # 接触判定的力阈值，根据你的传感器量程调整
    # =============================================================

    # 设备初始化
    sensor = TactileSensor()

    gripper = SO101ArmGripper()
    recorder = DataRecorder()

    print("连接设备...")
    if not sensor.connect():
        print("传感器连接失败")
        return
    if not gripper.connect():
        print("夹爪连接失败")
        return

    # 传感器数据队列初始化
    sensor.tactile_data_fifo = deque(maxlen=20)
    if sensor.start_cycle_read():
        print("触觉传感器采集线程启动成功")

    # 夹爪初始化
    gripper.wrist_roll()
    time.sleep(0.05)
    gripper.gripper_open()
    time.sleep(0.5)

    # 状态机定义
    state = "INIT_CONTACT"
    step_triggered = False
    current_input = 0.0  # 初始过盈量为0

    try:
        while True:
            # 读取最新触觉数据
            if len(sensor.tactile_data_fifo) == 0:
                time.sleep(0.01)
                continue
            latest_frame = sensor.tactile_data_fifo[-1]

            # 计算左右指法向力（和你原代码逻辑完全一致）
            a_force = latest_frame[:156]  # 拇指
            b_force = latest_frame[156:]  # 中指
            fa = np.sum(a_force[2::3])  # 法向力取Z轴分量，根据你的传感器坐标系调整
            fb = np.sum(b_force[2::3])  # 法向力取Z轴分量

            # =============================
            # 模型辨识专用状态机
            # =============================
            # 1. 初始接触：闭合夹爪直到有基础接触力
            if state == "INIT_CONTACT":
                print("[状态] 正在闭合夹爪，建立基础接触...")
                gripper.gripper_close()
                
                # 接触判定
                if max(fa, fb) > CONTACT_FORCE_THRESHOLD:
                    print("[状态] 检测到接触，停止闭合")
                    gripper.gripper_stop()
                    state = "STEADY_WAIT"
                    steady_wait_start = time.time()

            # 2. 等待系统稳定，避免接触瞬态干扰
            elif state == "STEADY_WAIT":
                wait_time = time.time() - steady_wait_start
                if wait_time > STEADY_WAIT_BEFORE_STEP:
                    print("[状态] 系统已稳定，开始记录数据，准备施加阶跃")
                    recorder.start()  # 启动时间记录
                    state = "STEP_TEST"
                else:
                    print(f"[状态] 等待稳定中... {wait_time:.1f}s / {STEADY_WAIT_BEFORE_STEP}s")

            # 3. 施加阶跃并全程记录数据
            elif state == "STEP_TEST":
                # 记录当前数据（全程记录，包括阶跃前、阶跃中、阶跃后）
                recorder.record(current_input, fa, fb)

                # 一次性触发阶跃指令（仅执行一次）
                if not step_triggered:
                    print(f"[状态] 施加阶跃指令！额外闭合 {STEP_AMPLITUDE}°")
                    # 执行阶跃闭合：根据你的gripper库调整，保证是相对位置增量
                    gripper.gripper_close()
                    # 开环时间控制：根据你的夹爪速度调整，保证完成STEP_AMPLITUDE的行程
                    time.sleep(STEP_AMPLITUDE * 0.1)
                    gripper.gripper_stop()
                    
                    current_input = STEP_AMPLITUDE  # 更新输入记录
                    step_triggered = True
                    step_start_time = time.time()

                # 阶跃后记录足够时间，直到力达到稳态
                if step_triggered and (time.time() - step_start_time > RECORD_AFTER_STEP):
                    print("[状态] 数据采集完成，正在保存...")
                    recorder.save_csv()
                    break

                # 实时打印数据
                if len(recorder.data['time']) > 0:
                    t_now = recorder.data['time'][-1]
                    f_now = recorder.data['output_force_total'][-1]
                    print(f"[记录] t: {t_now:.3f}s | 输入: {current_input}° | 总法向力: {f_now:.2f}")

            time.sleep(0.01)  # 采样频率100Hz，保证动态特性记录精度

    except KeyboardInterrupt:
        print("用户手动终止实验")
    except Exception as e:
        print(f"实验发生错误: {e}")
    finally:
        print("释放设备资源")
        # 异常终止也保存已采集的数据
        if 'recorder' in locals() and len(recorder.data['time']) > 0:
            recorder.save_csv("alpha_to_force_interrupted.csv")
        
        # 安全释放设备
        sensor.cycle_read_running = False
        time.sleep(0.5)
        gripper.gripper_stop()
        gripper.gripper_open()
        gripper.disconnect()
        sensor.disconnect()
        print("实验程序结束")

if __name__ == "__main__":
    main()
