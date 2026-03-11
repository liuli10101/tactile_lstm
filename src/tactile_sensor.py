#!/usr/bin/env python
# -*- coding: utf-8 -*-
import serial
import time
import threading
import serial.tools.list_ports
from typing import Optional, Dict, List, Tuple
from collections import deque


class TactileSensor:
    """
    触觉传感器高速通信集成板控制类（无UI版）
    核心功能：串口通信、传感器状态检测、分布力数据读取、循环读取控制
    对外接口：连接/断开、检查传感器状态、读取分布力、循环读/停等
    """

    def __init__(self, serial_port: str = "", baudrate: int = 921600):
        """
        初始化传感器控制器
        :param serial_port: 串口端口（如"/dev/ttyACM0"或"COM3"）
        :param baudrate: 波特率（默认921600）
        """
        # 串口配置
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None

        # 线程控制标记（循环读取）
        self.auto_receive_running = False
        self.cycle_read_running = False
        self.module_cycle_read_running = False
        self.cycle_read_thread: Optional[threading.Thread] = None
        self.module_cycle_thread: Optional[threading.Thread] = None
        self.read_counts = 0   #循环读取次数

        # 数据缓存
        self.tactile_data_fifo = deque(maxlen=20)  # 容量20的触觉数据缓冲区
        self.all_sensor_data = []  # 临时存储单轮传感器数据
        self.distribution_points_cache = {}  # 传感器分布力点数缓存
        self.connected_sensors = []  # 已连接的传感器名称列表
        self.maxz = 0

        # 传感器配置参数
        self.force_data_bytes = 6  # 合力数据长度（字节）
        self.distribution_data_bytes = 204  # 分布力数据长度（字节）
        self.total_data_bytes = self.force_data_bytes + self.distribution_data_bytes

        # 传感器模组配置 (地址0500-05A7)
        self.module_count = 28  # 28个传感器模组
        self.module_force_bytes = 6  # 每个模组合力数据: 3轴×2字节
        self.module_total_bytes = self.module_count * self.module_force_bytes  # 168字节

        # 掌心传感器特殊配置 - 只解析前9个分布力点
        self.palm_sensor_limit = 9

        # 传感器模组名称顺序
        self.module_names = [
            # 大拇指
            "大拇指近节", "大拇指中节", "大拇指指尖", "大拇指指甲",
            # 食指
            "食指近节", "食指中节", "食指指尖", "食指指甲",
            # 中指
            "中指近节", "中指中节", "中指指尖", "中指指甲",
            # 无名指
            "无名指近节", "无名指中节", "无名指指尖", "无名指指甲",
            # 小拇指
            "小拇指近节", "小拇指中节", "小拇指指尖", "小拇指指甲",
            # 掌心
            "掌心1", "掌心2", "掌心3", "掌心4",
            "掌心5", "掌心6", "掌心7", "掌心8"
        ]

        # 传感器分布力地址区间映射
        self.distribution_addr_ranges = {
            # 大拇指
            (0x1000, 0x11FF): "大拇指近节",
            (0x1200, 0x13FF): "大拇指中节",
            (0x1400, 0x15FF): "大拇指指尖",
            (0x1600, 0x17FF): "大拇指指甲",
            # 食指
            (0x1800, 0x19FF): "食指近节",
            (0x1A00, 0x1BFF): "食指中节",
            (0x1C00, 0x1DFF): "食指指尖",
            (0x1E00, 0x1FFF): "食指指甲",
            # 中指
            (0x2000, 0x21FF): "中指近节",
            (0x2200, 0x23FF): "中指中节",
            (0x2400, 0x25FF): "中指指尖",
            (0x2600, 0x27FF): "中指指甲",
            # 无名指
            (0x2800, 0x29FF): "无名指近节",
            (0x2A00, 0x2BFF): "无名指中节",
            (0x2C00, 0x2DFF): "无名指指尖",
            (0x2E00, 0x2FFF): "无名指指甲",
            # 小拇指
            (0x3000, 0x31FF): "小拇指近节",
            (0x3200, 0x33FF): "小拇指中节",
            (0x3400, 0x35FF): "小拇指指尖",
            (0x3600, 0x37FF): "小拇指指甲",
            # 掌心
            (0x3800, 0x38FF): "掌心1",
            (0x3900, 0x39FF): "掌心2",
            (0x3A00, 0x3AFF): "掌心3",
            (0x3B00, 0x3BFF): "掌心4",
            (0x3C00, 0x3CFF): "掌心5",
            (0x3D00, 0x3DFF): "掌心6",
            (0x3E00, 0x3EFF): "掌心7",
            (0x3F00, 0x3FFF): "掌心8"
        }

        # 传感器模组分布力点数地址映射
        self.distribution_points_addrs = {
            # 大拇指
            "大拇指近节": 0x0030,
            "大拇指中节": 0x0032,
            "大拇指指尖": 0x0034,
            "大拇指指甲": 0x0036,
            # 食指
            "食指近节": 0x0038,
            "食指中节": 0x003A,
            "食指指尖": 0x003C,
            "食指指甲": 0x003E,
            # 中指
            "中指近节": 0x0040,
            "中指中节": 0x0042,
            "中指指尖": 0x0044,
            "中指指甲": 0x0046,
            # 无名指
            "无名指近节": 0x0048,
            "无名指中节": 0x004A,
            "无名指指尖": 0x004C,
            "无名指指甲": 0x004E,
            # 小拇指
            "小拇指近节": 0x0050,
            "小拇指中节": 0x0052,
            "小拇指指尖": 0x0054,
            "小拇指指甲": 0x0056,
            # 掌心
            "掌心8": 0x0058,
            "掌心7": 0x005A,
            "掌心6": 0x005C,
            "掌心5": 0x005E,
            "掌心4": 0x0060,
            "掌心3": 0x0062,
            "掌心2": 0x0064,
            "掌心1": 0x0066
        }

        # 传感器状态地址定义
        self.sensor_status_addrs = {
            0x0010: "大拇指和食指传感器",
            0x0011: "中指和无名指传感器",
            0x0012: "小拇指和掌心1-4传感器",
            0x0013: "掌心5-8传感器"
        }

        # 系统控制地址
        self.SYSTEM_RESET_ADDR = 0x0022  # 高速通信集成板重启控制地址

        # 读取参数（替代原UI的var变量）
        self.read_addr = "1A00"
        self.read_len = "204"
        self.write_addr = "0017"
        self.write_data = "01"

    def log(self, message: str):
        """日志输出（控制台打印，可自行扩展为文件日志）"""
        pass
        # print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def list_serial_ports(self) -> List[str]:
        """列出可用的串口端口"""
        return [port.device for port in serial.tools.list_ports.comports()]

    def connect(self) -> bool:
        """
        连接传感器设备
        :return: 连接成功返回True，失败返回False
        """
        if self.ser and self.ser.is_open:
            self.log("已连接设备，无需重复连接")
            return True

        # 自动选择串口（如果未指定）
        if not self.serial_port:
            ports = self.list_serial_ports()
            if not ports:
                self.log("未检测到可用串口")
                return False
            self.serial_port = ports[0]
            self.log(f"自动选择串口: {self.serial_port}")

        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                write_timeout=0.1
            )
            if self.ser.is_open:
                self.log(f"成功连接到 {self.serial_port}，波特率 {self.baudrate}")
                return True
            return False
        except Exception as e:
            self.log(f"连接失败: {str(e)}")
            return False

    def disconnect(self) -> bool:
        """
        断开传感器连接
        :return: 断开成功返回True，失败返回False
        """
        # 停止所有循环读取
        self.stop_cycle_read()
        self.stop_module_cycle_read()
        self.auto_receive_running = False

        if self.ser and self.ser.is_open:
            self.ser.close()
            self.log("已断开设备连接")
            return True
        else:
            self.log("未连接设备，无需断开")
            return False

    def calculate_lrc(self, data: bytes) -> int:
        """计算LRC校验值"""
        lrc = 0
        for byte in data:
            lrc = (lrc + byte) & 0xFF
        lrc = ((~lrc) + 1) & 0xFF
        return lrc

    def build_request_frame(self, func_code: int, reg_addr: int, data: bytes = b"") -> bytes:
        """构建请求帧"""
        head = b"\x55\xAA"
        reserved = b"\x00"
        reg_addr_bytes = reg_addr.to_bytes(2, byteorder='little', signed=False)
        data_len = len(data) if func_code == 0x10 else int(self.read_len)
        data_len_bytes = data_len.to_bytes(2, byteorder='little', signed=False)
        frame_body = head + reserved + bytes([func_code]) + reg_addr_bytes + data_len_bytes + data
        lrc = self.calculate_lrc(frame_body)
        return frame_body + bytes([lrc])

    def parse_response_frame(self, frame: bytes) -> Optional[Dict]:
        """解析响应帧"""
        if len(frame) < 8 or frame[:2] != b"\xAA\x55":
            return None

        frame_body = frame[:-1]
        lrc = frame[-1]
        if self.calculate_lrc(frame_body) != lrc:
            self.log(f"LRC校验失败: 计算值={self.calculate_lrc(frame_body):02X}, 实际值={lrc:02X}")
            return None

        return {
            "reserved": frame[2],
            "func_code": frame[3],
            "reg_addr": int.from_bytes(frame[4:6], byteorder='little'),
            "data_len": int.from_bytes(frame[6:8], byteorder='little'),
            "data": frame[8:-1]
        }

    def check_sensor_status(self) -> List[str]:
        """
        检查传感器连接状态
        :return: 已连接的传感器名称列表
        """
        self.connected_sensors.clear()
        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return []

        self.log("===== 开始检查传感器连接状态 =====")

        # 读取所有状态地址
        for addr in self.sensor_status_addrs:
            try:
                self.read_addr = f"{addr:04X}"
                self.read_len = "1"
                reg_addr = addr
                read_len = 1

                request_frame = self.build_request_frame(0x03, reg_addr, b"")
                self.ser.write(request_frame)
                time.sleep(0.05)
                response = self.ser.read(128)

                if not response:
                    self.log(f"未收到地址{reg_addr:04X}的响应")
                    continue

                parsed = self.parse_response_frame(response)
                if parsed and parsed["func_code"] == 0x03 and parsed["data_len"] == 1:
                    status_byte = parsed["data"][0]
                    self.log(f"地址{reg_addr:04X}状态字节: 0x{status_byte:02X} ({bin(status_byte)[2:].zfill(8)})")

                    # 解析不同地址的状态
                    if addr == 0x0010:
                        self._parse_addr_0010(status_byte)
                    elif addr == 0x0011:
                        self._parse_addr_0011(status_byte)
                    elif addr == 0x0012:
                        self._parse_addr_0012(status_byte)
                    elif addr == 0x0013:
                        self._parse_addr_0013(status_byte)
            except Exception as e:
                self.log(f"读取地址{reg_addr:04X}错误: {str(e)}")

        self.log(f"传感器连接状态检查完成，共检测到 {len(self.connected_sensors)} 个连接成功的传感器")
        return self.connected_sensors

    def _parse_addr_0010(self, status_byte):
        """解析地址0010的状态（大拇指和食指）"""
        thumb_sensors = ["大拇指近节", "大拇指中节", "大拇指指尖", "大拇指指甲"]
        index_sensors = ["食指近节", "食指中节", "食指指尖", "食指指甲"]

        self.log("  大拇指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << i)) else "未连接"
            self.log(f"    {thumb_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(thumb_sensors[i])

        self.log("  食指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << (i + 4))) else "未连接"
            self.log(f"    {index_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(index_sensors[i])

    def _parse_addr_0011(self, status_byte):
        """解析地址0011的状态（中指和无名指）"""
        middle_sensors = ["中指近节", "中指中节", "中指指尖", "中指指甲"]
        ring_sensors = ["无名指近节", "无名指中节", "无名指指尖", "无名指指甲"]

        self.log("  中指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << i)) else "未连接"
            self.log(f"    {middle_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(middle_sensors[i])

        self.log("  无名指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << (i + 4))) else "未连接"
            self.log(f"    {ring_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(ring_sensors[i])

    def _parse_addr_0012(self, status_byte):
        """解析地址0012的状态（小拇指和掌心1-4）"""
        pinky_sensors = ["小拇指近节", "小拇指中节", "小拇指指尖", "小拇指指甲"]
        palm_sensors = ["掌心1", "掌心2", "掌心3", "掌心4"]

        self.log("  小拇指传感器状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << i)) else "未连接"
            self.log(f"    {pinky_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(pinky_sensors[i])

        self.log("  掌心传感器(1-4)状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << (i + 4))) else "未连接"
            self.log(f"    {palm_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(palm_sensors[i])

    def _parse_addr_0013(self, status_byte):
        """解析地址0013的状态（掌心5-8）"""
        palm_sensors = ["掌心5", "掌心6", "掌心7", "掌心8"]

        self.log("  掌心传感器(5-8)状态:")
        for i in range(4):
            status = "连接成功" if (status_byte & (1 << i)) else "未连接"
            self.log(f"    {palm_sensors[i]}: {status}")
            if status == "连接成功":
                self.connected_sensors.append(palm_sensors[i])

    def check_distribution_points(self) -> Dict[str, int]:
        """
        检查所有传感器的分布力点数
        :return: 传感器名称到点数的映射字典
        """
        self.distribution_points_cache.clear()
        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return {}

        self.log("===== 开始检查分布力点数 =====")

        # 按类别分组读取
        groups = {
            "大拇指": ["大拇指近节", "大拇指中节", "大拇指指尖", "大拇指指甲"],
            "食指": ["食指近节", "食指中节", "食指指尖", "食指指甲"],
            "中指": ["中指近节", "中指中节", "中指指尖", "中指指甲"],
            "无名指": ["无名指近节", "无名指中节", "无名指指尖", "无名指指甲"],
            "小拇指": ["小拇指近节", "小拇指中节", "小拇指指尖", "小拇指指甲"],
            "掌心": ["掌心1", "掌心2", "掌心3", "掌心4", "掌心5", "掌心6", "掌心7", "掌心8"]
        }

        for group_name, modules in groups.items():
            self.log(f"\n===== {group_name}传感器分布力点数 =====")
            for module in modules:
                if module not in self.distribution_points_addrs:
                    self.log(f"  {module}: 无对应地址信息")
                    self.distribution_points_cache[module] = 0
                    continue

                addr = self.distribution_points_addrs[module]
                try:
                    self.read_addr = f"{addr:04X}"
                    self.read_len = "2"
                    reg_addr = addr
                    read_len = 2

                    request_frame = self.build_request_frame(0x03, reg_addr, b"")
                    self.ser.write(request_frame)
                    time.sleep(0.05)
                    response = self.ser.read(128)

                    if not response:
                        self.log(f"  {module} (地址{reg_addr:04X}): 未收到响应")
                        self.distribution_points_cache[module] = 0
                        continue

                    parsed = self.parse_response_frame(response)
                    if parsed and parsed["func_code"] == 0x03 and parsed["data_len"] == 2:
                        point_count = int.from_bytes(parsed["data"], byteorder='little', signed=False)
                        self.distribution_points_cache[module] = point_count

                        byte_count = point_count * 3
                        if module.startswith("掌心"):
                            actual_points = min(point_count, self.palm_sensor_limit)
                            status = f"已连接 (将只解析前{actual_points}个点)" if point_count > 0 else "未连接"
                        else:
                            status = "已连接" if point_count > 0 else "未连接"

                        self.log(
                            f"  {module} (地址{reg_addr:04X}): "
                            f"点数={point_count}, 字节数={byte_count}, 状态={status}"
                        )
                    else:
                        self.log(f"  {module} (地址{reg_addr:04X}): 读取失败")
                        self.distribution_points_cache[module] = 0
                except Exception as e:
                    self.log(f"  {module} (地址{reg_addr:04X}) 错误: {str(e)}")
                    self.distribution_points_cache[module] = 0

        self.log("\n===== 分布力点数检查完成 =====")
        return self.distribution_points_cache

    def get_address_by_sensor_name(self, sensor_name: str) -> Tuple[Optional[int], Optional[int]]:
        """根据传感器名称获取对应的地址范围"""
        for (start, end), name in self.distribution_addr_ranges.items():
            if name == sensor_name:
                return (start, end)
        return (None, None)

    def parse_normal_force_data(self, data: bytes, addr: int, source: str = "") -> Tuple[List[Dict], List[float],float]:
        """解析分布力数据并返回结构化数据和扁平化数据"""
        # 根据地址获取传感器名称
        sensor_name = None
        for (start, end), name in self.distribution_addr_ranges.items():
            if start <= addr <= end:
                sensor_name = name
                break

        if not sensor_name:
            sensor_name = f"未知传感器(0x{addr:04X})"

        parsed = []
        one_sensor_data = []
        total_groups = len(data) // 3
        remainder = len(data) % 3

        if remainder != 0:
            self.log(f"{source}{sensor_name}分布力数据长度不是3的倍数，剩余{remainder}字节将被忽略")

        # 掌心传感器特殊处理
        if sensor_name.startswith("掌心"):
            total_groups = min(total_groups, self.palm_sensor_limit)

        for i in range(total_groups):
            offset = i * 3
            b1, b2, b3 = data[offset], data[offset + 1], data[offset + 2]

            val1 = b1 if b1 <= 127 else b1 - 256
            val2 = b2 if b2 <= 127 else b2 - 256
            val3 = b3

            scaled1 = round(val1 * 0.1, 1)
            scaled2 = round(val2 * 0.1, 1)
            scaled3 = round(val3 * 0.1, 1)

            one_sensor_data.append(scaled1)
            one_sensor_data.append(scaled2)
            one_sensor_data.append(scaled3)

            parsed.append({
                "index": i,
                "raw_bytes": (b1, b2, b3),
                "raw_hex": (f"0x{b1:02X}", f"0x{b2:02X}", f"0x{b3:02X}"),
                "converted": (val1, val2, val3),
                "scaled": (scaled1, scaled2, scaled3)
            })

        if parsed:
            self.log(f"\n===== {source}{sensor_name}分布力数据 =====")
            self.log(f"共{len(parsed)}组数据（每组×0.1N）")

            # 计算最大值
            max_x = max(p["scaled"][0] for p in parsed)
            max_y = max(p["scaled"][1] for p in parsed)
            max_z = max(p["scaled"][2] for p in parsed)

            max_x_idx = next(i for i, p in enumerate(parsed) if p["scaled"][0] == max_x)
            max_y_idx = next(i for i, p in enumerate(parsed) if p["scaled"][1] == max_y)
            max_z_idx = next(i for i, p in enumerate(parsed) if p["scaled"][2] == max_z)

            self.log(f"X轴最大值: {max_x}N (组{max_x_idx})")
            self.log(f"Y轴最大值: {max_y}N (组{max_y_idx})")
            self.log(f"Z轴最大值: {max_z}N (组{max_z_idx})")

        return parsed, one_sensor_data, max_z

    def read_connected_sensors(self) -> Tuple[bool, List[float]]:
        """
        读取已连接传感器的分布力数据
        :return: (是否成功, 扁平化的传感器数据列表)
        """
        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return False, []

        # 先检查传感器状态
        if not self.connected_sensors:
            self.log("没有检测到连接成功的传感器")
            self.check_sensor_status()
            return False, []

        self.log("\n===== 开始读取所有连接成功的传感器分布力 =====")
        self.log(f"共检测到 {len(self.connected_sensors)} 个连接成功的传感器")

        # 按类别分组
        groups = {
            "大拇指": [], "食指": [], "中指": [],
            "无名指": [], "小拇指": [], "掌心": []
        }

        for sensor in self.connected_sensors:
            if sensor.startswith("大拇指"):
                groups["大拇指"].append(sensor)
            elif sensor.startswith("食指"):
                groups["食指"].append(sensor)
            elif sensor.startswith("中指"):
                groups["中指"].append(sensor)
            elif sensor.startswith("无名指"):
                groups["无名指"].append(sensor)
            elif sensor.startswith("小拇指"):
                groups["小拇指"].append(sensor)
            elif sensor.startswith("掌心"):
                groups["掌心"].append(sensor)

        success_count = 0
        fail_count = 0
        all_flat_data = []

        # 逐个读取传感器数据
        for group_name, sensors in groups.items():
            if sensors:
                self.log(f"\n----- {group_name}传感器 -----")
                for idx, sensor in enumerate(sensors):
                    self.log(f"\n[{idx + 1}/{len(sensors)}] 处理 {sensor}...")

                    # 获取地址范围
                    start_addr, end_addr = self.get_address_by_sensor_name(sensor)
                    if not start_addr or not end_addr:
                        self.log(f"  {sensor}: 未找到对应的地址范围")
                        fail_count += 1
                        continue

                    # 获取点数地址
                    points_addr = self.distribution_points_addrs.get(sensor)
                    if not points_addr:
                        self.log(f"  {sensor}: 未找到点数地址信息")
                        fail_count += 1
                        continue

                    try:
                        # 读取点数
                        self.read_addr = f"{points_addr:04X}"
                        self.read_len = "2"
                        request_frame = self.build_request_frame(0x03, points_addr, b"")
                        self.ser.write(request_frame)
                        time.sleep(0.05)
                        response = self.ser.read(128)

                        if not response:
                            self.log(f"  {sensor}: 未收到点数响应")
                            fail_count += 1
                            continue

                        parsed = self.parse_response_frame(response)
                        if parsed and parsed["func_code"] == 0x03 and parsed["data_len"] == 2:
                            point_count = int.from_bytes(parsed["data"], byteorder='little', signed=False)

                            # 掌心传感器特殊处理
                            if sensor.startswith("掌心"):
                                actual_points = min(point_count, self.palm_sensor_limit)
                                self.log(f"  {sensor}: 原始点数={point_count}, 将只解析前{actual_points}个点")
                                data_len = actual_points * 3
                            else:
                                data_len = point_count * 3

                            if data_len <= 0:
                                self.log(f"  {sensor}: 无效的点数({point_count})")
                                fail_count += 1
                                continue

                            # 读取分布力数据
                            self.read_addr = f"{start_addr:04X}"
                            self.read_len = str(data_len)
                            request_frame = self.build_request_frame(0x03, start_addr, b"")
                            self.ser.write(request_frame)
                            time.sleep(0.05)
                            data_response = self.ser.read(1024)

                            if not data_response:
                                self.log(f"  {sensor}: 未收到分布力数据")
                                fail_count += 1
                                continue

                            data_parsed = self.parse_response_frame(data_response)
                            if data_parsed and data_parsed["func_code"] == 0x03:
                                self.log(f"  成功读取 {sensor} 分布力数据 (地址:0x{start_addr:04X}, 长度:{data_len}字节)")
                                _, sensor_flat_data,self.maxz = self.parse_normal_force_data(data_parsed["data"], start_addr, source=f"[{sensor}] ")
                                all_flat_data.extend(sensor_flat_data)
                                success_count += 1
                            else:
                                self.log(f"  {sensor}: 分布力数据解析失败")
                                fail_count += 1
                        else:
                            self.log(f"  {sensor}: 点数读取失败")
                            fail_count += 1
                    except Exception as e:
                        self.log(f"  {sensor} 读取错误: {str(e)}")
                        fail_count += 1

        self.log("\n===== 传感器分布力读取完成 =====")
        self.log(f"读取结果: 成功 {success_count} 个, 失败 {fail_count} 个")

        # 更新数据缓存
        self.all_sensor_data = all_flat_data
        self.tactile_data_fifo.append(all_flat_data)

        return success_count > 0, all_flat_data

    def read_registers(self) -> Tuple[bool, List[float]]:
        """单次读取寄存器（分布力数据），内部调用read_connected_sensors"""
        return self.read_connected_sensors()

    def _cycle_read_worker(self):
        """循环读取工作线程"""
        self.log("循环读取线程已启动")
        while self.cycle_read_running:
            try:
                self.read_connected_sensors()
                self.read_counts += 1
                time.sleep(0.05)  # 读取间隔（可调整）
            except Exception as e:
                self.log(f"循环读取异常: {str(e)}")
                time.sleep(0.1)
        self.log("循环读取线程已停止")

    def start_cycle_read(self) -> bool:
        """
        开始循环读取传感器分布力数据
        :return: 启动成功返回True，失败返回False
        """
        if self.cycle_read_running:
            self.log("已在循环读取中")
            return False

        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return False

        self.cycle_read_running = True
        self.cycle_read_thread = threading.Thread(target=self._cycle_read_worker, daemon=True)
        self.cycle_read_thread.start()
        self.log("开始循环读取传感器分布力数据...")
        return True

    def stop_cycle_read(self) -> bool:
        """
        停止循环读取
        :return: 停止成功返回True，失败返回False
        """
        if not self.cycle_read_running:
            self.log("未在循环读取中")
            return False

        self.cycle_read_running = False
        # 等待线程结束
        if self.cycle_read_thread and self.cycle_read_thread.is_alive():
            self.cycle_read_thread.join(timeout=1.0)
        self.log("已停止循环读取传感器分布力数据")
        return True

    def read_module_forces(self) -> Tuple[bool, List[Dict]]:
        """
        读取传感器模组合力数据
        :return: (是否成功, 解析后的模组合力数据列表)
        """
        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return False, []

        try:
            self.read_addr = "0500"
            self.read_len = str(self.module_total_bytes)
            reg_addr = int(self.read_addr, 16)
            read_len = int(self.read_len)

            request_frame = self.build_request_frame(0x03, reg_addr, b"")
            self.ser.write(request_frame)
            self.log(f"发送传感器模组合力请求: 地址={reg_addr:04X}, 长度={read_len}, 帧={request_frame.hex()}")

            time.sleep(0.01)
            response = self.ser.read(1024)

            if not response:
                self.log("未收到传感器模组合力响应")
                return False, []

            parsed = self.parse_response_frame(response)
            if parsed and parsed["func_code"] == 0x03:
                self.log(f"传感器模组合力读取成功: 地址={parsed['reg_addr']:04X}, 长度={parsed['data_len']}")
                module_data = self.parse_module_forces(parsed["data"])
                return True, module_data
            else:
                self.log("传感器模组合力读取失败")
                return False, []
        except Exception as e:
            self.log(f"读取模组合力错误: {str(e)}")
            return False, []

    def parse_module_forces(self, data: bytes) -> List[Dict]:
        """解析传感器模组合力数据"""
        parsed = []
        if len(data) != self.module_total_bytes:
            self.log(f"警告: 传感器模组合力数据长度不符 - 预期: {self.module_total_bytes}字节, 实际: {len(data)}字节")

        valid_modules = min(len(data) // self.module_force_bytes, self.module_count)
        self.log(f"传感器模组合力数据: 共{valid_modules}/{self.module_count}个有效模组数据（仅解析低字节）")

        for i in range(valid_modules):
            offset = i * self.module_force_bytes
            if offset + self.module_force_bytes > len(data):
                self.log("警告: 数据长度不足，终止解析")
                break

            # 提取低字节
            fx_low_byte = data[offset]
            fy_low_byte = data[offset + 2]
            fz_low_byte = data[offset + 4]

            # 完整2字节
            fx_bytes = data[offset:offset + 2]
            fy_bytes = data[offset + 2:offset + 4]
            fz_bytes = data[offset + 4:offset + 6]

            # 转换值
            fx_raw = fx_low_byte if fx_low_byte <= 127 else fx_low_byte - 256
            fy_raw = fy_low_byte if fy_low_byte <= 127 else fy_low_byte - 256
            fz_raw = fz_low_byte

            # 物理值
            fx_scaled = round(fx_raw * 0.1, 1)
            fy_scaled = round(fy_raw * 0.1, 1)
            fz_scaled = round(fz_raw * 0.1, 1)

            parsed.append({
                "index": i,
                "name": self.module_names[i] if i < len(self.module_names) else f"未知模组{i}",
                "raw_hex": (
                    fx_bytes.hex().upper(),
                    fy_bytes.hex().upper(),
                    fz_bytes.hex().upper()
                ),
                "used_byte": (
                    f"0x{fx_low_byte:02X}",
                    f"0x{fy_low_byte:02X}",
                    f"0x{fz_low_byte:02X}"
                ),
                "converted": (fx_raw, fy_raw, fz_raw),
                "scaled": (fx_scaled, fy_scaled, fz_scaled)
            })

        # 打印模组数据
        if parsed:
            self.log("\n===== 传感器模组合力数据 =====")
            for module in parsed:
                self.log(f"{module['name']:8s} | Fx={module['scaled'][0]:5.1f}N Fy={module['scaled'][1]:5.1f}N Fz={module['scaled'][2]:5.1f}N")

        return parsed

    def _module_cycle_worker(self):
        """模组循环读取工作线程"""
        self.log("模组循环读取线程已启动")
        while self.module_cycle_read_running:
            try:
                self.read_module_forces()
                time.sleep(0.05)
            except Exception as e:
                self.log(f"模组循环读取异常: {str(e)}")
                time.sleep(0.1)
        self.log("模组循环读取线程已停止")

    def start_module_cycle_read(self) -> bool:
        """
        开始循环读取传感器模组合力
        :return: 启动成功返回True，失败返回False
        """
        if self.module_cycle_read_running:
            self.log("已在循环读取传感器模组数据中")
            return False

        if not self.ser or not self.ser.is_open:
            self.log("请先连接设备")
            return False

        self.module_cycle_read_running = True
        self.module_cycle_thread = threading.Thread(target=self._module_cycle_worker, daemon=True)
        self.module_cycle_thread.start()
        self.log("开始循环读取传感器模组数据...")
        return True

    def stop_module_cycle_read(self) -> bool:
        """
        停止循环读取传感器模组合力
        :return: 停止成功返回True，失败返回False
        """
        if not self.module_cycle_read_running:
            self.log("未在循环读取传感器模组数据中")
            return False

        self.module_cycle_read_running = False
        if self.module_cycle_thread and self.module_cycle_thread.is_alive():
            self.module_cycle_thread.join(timeout=1.0)
        self.log("已停止循环读取传感器模组数据")
        return True

    def get_tactile_data(self) -> deque:
        """
        获取缓存的触觉数据
        :return: 数据缓冲区（deque）
        """
        return self.tactile_data_fifo

    def get_latest_tactile_data(self) -> Optional[List[float]]:
        """
        获取最新的触觉数据
        :return: 最新的扁平化数据列表，无数据返回None
        """
        if self.tactile_data_fifo:
            return self.tactile_data_fifo[-1]
        return None


