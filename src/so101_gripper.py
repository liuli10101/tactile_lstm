#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
SO101从动臂夹爪控制类模块（非阻塞闭合版）
核心特性：
1. 夹爪闭合为非阻塞函数，支持实时中断
2. 对外接口：gripper_open/gripper_close（非阻塞）/gripper_stop
3. 闭合过程中可通过gripper_stop即时中断
"""
import time
import threading
import numpy as np
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


class SO101ArmGripper:
    """SO101从动臂夹爪控制类（非阻塞闭合版）"""

    def __init__(self, serial_port="/dev/ttyACM0", max_relative_move=100.0, use_degrees=False):
        """
        初始化夹爪控制器
        :param serial_port: 串口端口（Linux/Mac: "/dev/ttyUSB0"，Windows: "COM3"）
        :param max_relative_move: 单次最大移动量（-100~100，安全限位）
        :param use_degrees: 是否使用角度模式（False：归一化0~100，True：角度值）
        """
        # 配置参数
        self.serial_port = serial_port
        self.max_relative_move = max_relative_move
        self.use_degrees = use_degrees

        # 核心实例与状态
        self.config = None          # 配置类实例
        self.arm = None             # 从动臂控制器实例
        self.is_connected = False   # 连接状态标记

        # 非阻塞闭合相关（核心新增）
        self.close_thread = None    # 闭合动作线程
        self.is_closing = False     # 是否正在闭合（非阻塞状态标记）
        self.arrive = False         # 闭合到达标志
        self.stop_close_flag = False# 闭合停止标志位（用于中断）
        self.close_step = 2.0       # 单次闭合步长（越小中断越灵敏，建议1~5）
        self.close_interval = 0.05  # 单次步长执行间隔（秒，越小闭合越平滑）

        self.t = 0
        self.Rab = np.array([[np.cos(self.t), 0, -np.sin(self.t)],
                            [0, -1, 0],
                            [-np.sin(self.t), 0, -np.cos(self.t)]])
        self.APBorg = np.array([73.03*np.cos(self.t)-78.54, 0, -73.03*np.sin(self.t)-27.46])

        # 初始化配置
        self._init_config()


    def _init_config(self):
        """内部方法：初始化SO101配置类"""
        try:
            self.config = SO101FollowerConfig(
                port=self.serial_port,
                max_relative_target=self.max_relative_move,
                use_degrees=self.use_degrees,
                disable_torque_on_disconnect=True
            )
            print(f"[SO101夹爪] 配置初始化成功：串口={self.serial_port}")
        except Exception as e:
            raise Exception(f"[SO101夹爪] 配置初始化失败：{e}")

    def connect(self, calibrate=True):
        """连接从动臂并校准（首次使用必须调用）"""
        if self.is_connected:
            print("[SO101夹爪] 已连接，无需重复操作")
            return True

        try:
            self.arm = SO101Follower(self.config)
            self.arm.connect(calibrate=calibrate)
            self.is_connected = True
            print("[SO101夹爪] 连接并校准完成")
            return True
        except DeviceAlreadyConnectedError:
            print("[SO101夹爪] 错误：已被其他进程连接")
            return False
        except Exception as e:
            print(f"[SO101夹爪] 连接失败：{e}")
            return False

    def _get_current_gripper_pos(self):
        """内部方法：获取当前夹爪位置"""
        if not self.is_connected:
            raise DeviceNotConnectedError("未连接，无法获取夹爪位置")
        try:
            current_obs = self.arm.get_observation()
            return current_obs.get("gripper.pos", 50.0)  # 默认50.0防止空值
        except Exception as e:
            print(f"[SO101夹爪] 获取位置失败：{e}")
            return 50.0

    def _do_close(self, target_pos):
        """
        内部方法：实际执行夹爪闭合（线程执行体）
        :param target_pos: 闭合目标位置（0~100）
        """
        try:
            current_pos = self._get_current_gripper_pos()
            print(f"[SO101夹爪] 开始非阻塞闭合：当前位置={current_pos:.2f}，目标位置={target_pos:.2f}")

            # 逐步闭合循环（核心：每步检查停止标志位）
            while not self.stop_close_flag:
                # 计算下一步位置（向目标位置靠近）
                if current_pos > target_pos:
                    next_pos = max(current_pos - self.close_step, target_pos)
                else:
                    break  # 已到达目标位置，结束闭合

                # 发送单步闭合指令
                self.arm.send_action({"gripper.pos": next_pos})
                time.sleep(self.close_interval)  # 短间隔保证平滑闭合

                # 更新当前位置
                current_pos = self._get_current_gripper_pos()
                print(f"[SO101夹爪] 闭合中：当前位置={current_pos:.2f}", end="\r")

                # 检查是否到达目标位置
                if current_pos-1 <= target_pos :
                    self.arrive = True
                    break

        except Exception as e:
            print(f"\n[SO101夹爪] 闭合线程异常：{e}")
        finally:
            # 重置状态标记
            self.is_closing = False
            self.stop_close_flag = False
            print(f"\n[SO101夹爪] 闭合结束，最终位置={self._get_current_gripper_pos():.2f}")

    def gripper_close(self, target_pos=16.0):
        """
        【对外接口】非阻塞夹爪闭合（立即返回，后台线程执行）
        :param target_pos: 闭合目标位置（0~100，0为全闭）
        :return: 线程启动是否成功（bool）
        """
        if not self.is_connected:
            print("[SO101夹爪] 错误：未连接，无法执行闭合")
            return False

        if self.is_closing:
            print("[SO101夹爪] 警告：已在闭合过程中，请勿重复调用")
            return False

        # 重置停止标志位，启动闭合线程
        self.stop_close_flag = False
        self.is_closing = True
        self.close_thread = threading.Thread(
            target=self._do_close,
            args=(target_pos,),
            daemon=True  # 守护线程，主程序退出时自动结束
        )
        self.close_thread.start()
        return True

    def gripper_stop(self):
        """
        【对外接口】中断夹爪闭合（非阻塞）
        :return: 执行是否成功（bool）
        """
        if not self.is_connected:
            print("[SO101夹爪] 错误：未连接，无法执行停止")
            return False

        if not self.is_closing:
            print("[SO101夹爪] 警告：未在闭合过程中，无需停止")
            return True

        # 设置停止标志位，中断闭合线程
        self.stop_close_flag = True
        # 等待线程结束（最多等待1秒，防止卡死）
        if self.close_thread is not None and self.close_thread.is_alive():
            self.close_thread.join(timeout=1.0)

        # 发送当前位置指令，强制保持静止
        current_pos = self._get_current_gripper_pos()
        self.arm.send_action({"gripper.pos": current_pos})
        print(f"[SO101夹爪] 已中断闭合，停止位置={current_pos:.2f}")

        # 重置状态
        self.is_closing = False
        return True

    def gripper_open(self, open_pos=80.0, delay=0.5):
        """【对外接口】阻塞式夹爪打开（打开动作无需实时中断）"""
        if not self.is_connected:
            print("[SO101夹爪] 错误：未连接，无法执行打开")
            return False

        try:
            sent_action = self.arm.send_action({"gripper.pos": open_pos})
            time.sleep(delay)
            print(f"[SO101夹爪] 夹爪已打开至：{sent_action['gripper.pos']:.2f}")
            return True
        except Exception as e:
            print(f"[SO101夹爪] 打开失败：{e}")
            return False

    def disconnect(self):
        """安全断开连接"""
        # 先停止闭合（若正在闭合）
        if self.is_closing:
            self.gripper_stop()

        if not self.is_connected:
            print("[SO101夹爪] 未连接，无需断开")
            return True

        try:
            self.arm.disconnect()
            self.is_connected = False
            print("[SO101夹爪] 已安全断开，扭矩禁用")
            return True
        except Exception as e:
            print(f"[SO101夹爪] 断开失败：{e}")
            return False

    def force_balance (self,fa,fb):
        Afb =  self.Rab @ fb 
        dot_product = np.dot(fa, Afb)
        norm_a = np.linalg.norm(fa)
        norm_b = np.linalg.norm(Afb)
        cos_theta = dot_product / (norm_a * norm_b)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 范围限制
        # 求夹角
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        print ("夹角为",theta_deg,"度")
        if theta_deg < 174: 
            return False
        else :
            return True
