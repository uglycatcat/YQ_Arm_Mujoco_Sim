import keyboard
import time
import pygame
from threading import Thread, Event, Lock

class ArmController:
    def __init__(self):
        """初始化控制器"""
        # 定义键盘控制字典（映射关系）
        keyboard_dictionary=None
        # 定义xbox手柄控制字典（映射关系）
        xbox_dictionary=None

    def switch_mode(self,mode):
        """切换控制模式"""
        
    def start(self):
        """启动控制器线程"""
    
    def stop(self):
        """停止控制器线程"""
    
    def switch_controller(self,controller):
        """切换控制器（手柄/键盘）"""
    
    def handle_keyboard_input(self):
        """处理键盘输入"""
    
    def handle_xbox_input(self):
        """处理Xbox手柄输入"""
        
    def check_xbox_connection(self):
        """检查Xbox手柄连接状态"""

    def update():
        """更新接收数据"""

controller=ArmController()