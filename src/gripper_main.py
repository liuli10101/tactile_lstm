import time
# 导入你之前的传感器模块和SO101夹爪模块
from tactile_sensor import TactileSensor  # 传感器模块化类
from so101_gripper  import SO101ArmGripper# SO101夹爪模块化类

def main():
    """主程序：读取传感器值并控制夹爪闭合，传感器值>0时停止"""
    # 初始化设备对象（初始化为None，方便后续异常处理）
    sensor = None
    gripper = None

    try:
        # 1. 初始化传感器和夹爪
        print("正在初始化传感器和SO101夹爪...")
        sensor = TactileSensor()  # 实例化传感器对象（根据你的模块调整参数）
        gripper = SO101ArmGripper()  # 实例化夹爪对象（根据你的模块调整参数）
        
        # 2. 确认设备初始化成功
        if not sensor.connect() or not gripper.connect():
            raise RuntimeError("传感器或夹爪连接失败，请检查硬件连接")
        
        print("设备初始化完成，开始控制夹爪闭合...")
        
        # 3. 核心逻辑：发送闭合指令并循环检测传感器值
        # 先发送夹爪闭合指令
        gripper.gripper_open()  #打开夹爪
        gripper.gripper_close()  # 发送夹爪闭合指令
        print("已发送夹爪闭合指令，开始检测传感器值...")

        # 循环读取传感器值，直到满足停止条件
        while True:
            # 读取传感器数值（捕获单次读取失败的异常）
            try:
                sensor_value = sensor.read_connected_sensors()
                group=sensor.get_tactile_data()
                print(f"当前传感器值：{sensor_value}")
                #print(f"传感器值组：{group}")
            except Exception as read_err:
                print(f"传感器读取失败：{read_err}，将重试...")
                time.sleep(0.05)  # 读取失败时短暂延时后重试
                continue

            # 停止条件：传感器值>0时停止夹爪
            if sensor.maxz > 0:
                print(f"已夹取到物体，停止夹爪闭合...")
                gripper.gripper_stop()  # 发送夹爪停止指令
                break  # 退出循环
            elif gripper.arrive :
                print(f"夹爪闭合完毕")
                break
            # 控制读取频率（避免高频读取占用资源，可根据需求调整）
            time.sleep(0.02)  # 20ms读取一次，平衡响应速度和资源占用

    except RuntimeError as e:
        # 处理设备连接等严重错误
        print(f"程序运行错误：{e}")
        # 若夹爪已初始化，紧急停止夹爪
        if gripper:
            gripper.gripper_stop()
    except KeyboardInterrupt:
        # 处理用户手动终止程序（Ctrl+C）
        print("\n用户手动终止程序，正在停止夹爪...")
        if gripper:
            gripper.gripper_stop()
    finally:
        # 4. 释放设备资源（无论程序正常/异常结束，都执行）
        print("正在释放设备资源...")
        if sensor:
            sensor.disconnect()  # 关闭传感器连接
        if gripper:
            gripper.disconnect()  # 关闭夹爪连接
        print("程序结束，资源已释放")

if __name__ == "__main__":
    # 启动主程序
    main()
