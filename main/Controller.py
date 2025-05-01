# 控制器线程
# 该线程负责接收用户的手柄/键盘输入
# 对输入进行滤波，平滑等处理，转换为速度，位置信息。接收模式变化相关信号。
# 该线程应当是一个触发式线程，即在接收到输入后改变关键数据trans的值
# 当外部程序调用update函数时，返回当前的trans值
# 该线程应当是一个守护线程，主线程结束后自动结束
# 传递给main函数中内容
import keyboard
import time
import pygame
import numpy as np
from threading import Thread, Event, Lock

class ArmController:
    def __init__(self):
        """初始化控制器"""
        # 定义键盘控制字典（映射关系）
        self.keyboard_dictionary=None
        # 定义xbox手柄控制字典（映射关系）
        self.xbox_dictionary=None
        # 运动参数
        self.TRANS_STEP = 0.0001  # 平移步长 1cm（降低五倍）
        self.ROT_STEP = np.radians(0.05)  # 旋转步长 0.2度（降低五倍）
        # 添加手柄相关初始化
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.is_keyboard = True
        # 空格是否按下（用于切换模式）
        self.space_pressed = False
        # 手柄控制参数
        self.joystick_max_speed = 0.0001  # 最大移动速度
        self.joystick_filter_alpha = 0.01  # 平滑因子
        self.joystick_deadzone = 0.001      # 死区阈值
        # 键盘控制参数
        self.keyboard_max_speed = 0.0001 #最大移动速度
        self.keyboard_filter_alpha = 0.01  # 平滑因子
        # 控制模式参数(默认为13)
        self.control_mode=13
        # 控制数据（在update中的返回值）
        self.trans_data=None
        # 控制器线程参数
        self.thread = None
        self.lock = Lock()
        self.running = Event()
        

    def help(self):
        """打印控制信息"""
        print("""
        =============================
        机械臂控制器 使用说明
        =============================

        [ 键盘模式 按键控制 ]
        - 机械臂末端平移：
        W / S : 前进 / 后退
        A / D : 左移 / 右移
        Q / E : 上移 / 下移
        
        [ 手柄Xbox模式 摇杆和按键控制 ]
        - 机械臂末端平移：
        左摇杆Y轴: 前进 / 后退
        左摇杆X轴: 左移 / 右移
        右摇杆Y轴: 上移 / 下移
        A键 : 切换到模式15
        B键 : 切换到模式13
        
        [ 模式切换 ]
        - 空格键 : 切换输入模式（键盘 / 手柄）
        
        [ 注意事项 ]
        - 请确保手柄已连接并处于活动状态。
        - 程序启动时默认为手柄控制。
        - 程序启动时手柄未连接则切换到键盘控制。
        - 手柄在程序运行中连接,程序无法切换到手柄模式。

        请确保 MuJoCo 界面处于激活状态，否则键盘输入可能无效。
        """)
        
    def switch_mode(self, mode):
        """切换控制模式"""
        self.control_mode = mode
        print(f"已切换控制模式为: {mode}")
        
    def start(self):
        """启动控制器线程"""
        if not self.running.is_set():
            self.running.set()
            self.thread = Thread(target=self._controller_loop, daemon=True)
            self.thread.start()
            print("控制器线程已启动。")

    def stop(self):
        """停止控制器线程"""
        self.running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("控制器线程已停止。")
        
    def _controller_loop(self):
        """主控制循环"""
        while self.running.is_set():
            self.check_xbox_connection()

            # 检查是否按下空格来切换控制器
            if keyboard.is_pressed('space'):
                if not self.space_pressed:
                    self.space_pressed = True
                    if self.is_keyboard:
                        self.switch_controller("xbox")
                    else:
                        self.switch_controller("keyboard")
                    time.sleep(0.3)
            else:
                self.space_pressed = False

            # 获取当前控制器输入
            if self.is_keyboard:
                trans = self.handle_keyboard_input()
            else:
                trans = self.handle_xbox_input()

            with self.lock:
                self.trans_data = trans

            time.sleep(0.01)
    
    def switch_controller(self, controller):
        """切换控制器模式"""
        if controller == "keyboard":
            self.is_keyboard = True
            print("已切换到键盘模式")
        elif controller == "xbox":
            if self.joystick is not None:
                self.is_keyboard = False
                print("已切换到手柄模式")
            else:
                print("手柄未连接，无法切换到手柄模式")
                    
    def handle_keyboard_input(self):
        """处理键盘输入"""
        trans = np.zeros(3)
        if keyboard.is_pressed('s'): trans[0] += self.TRANS_STEP
        if keyboard.is_pressed('w'): trans[0] -= self.TRANS_STEP
        if keyboard.is_pressed('d'): trans[1] += self.TRANS_STEP
        if keyboard.is_pressed('a'): trans[1] -= self.TRANS_STEP
        if keyboard.is_pressed('UP'): trans[2] += self.TRANS_STEP
        if keyboard.is_pressed('DOWN'): trans[2] -= self.TRANS_STEP
        return trans

    def handle_xbox_input(self):
        """处理Xbox手柄输入"""
        if self.joystick is None:
            return np.zeros(3)
            
        pygame.event.pump()

        left_y = -self.apply_deadzone(self.joystick.get_axis(1))
        left_x = self.apply_deadzone(self.joystick.get_axis(0))
        right_y = -self.apply_deadzone(self.joystick.get_axis(3))

        if self.joystick.get_button(0):  # A键
            self.switch_mode(15)
        if self.joystick.get_button(1):  # B键
            self.switch_mode(13)

        return np.array([
            left_y * self.joystick_max_speed,
            left_x * self.joystick_max_speed,
            right_y * self.joystick_max_speed
        ])
        
    def check_xbox_connection(self):
        """检查Xbox手柄连接状态"""
        current_joystick_count = pygame.joystick.get_count()
        if current_joystick_count > 0:
            if self.joystick is None:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"手柄已连接：{self.joystick.get_name()}")
        else:
            if self.joystick is not None:
                print("手柄已断开，自动切换到键盘模式")
                self.joystick = None
                self.is_keyboard = True
                
    def update_data(self):
        """获取最新的trans数据"""
        with self.lock:
            return np.copy(self.trans_data) if self.trans_data is not None else np.zeros(3)
    
    def update_mode(self):
        
        return self.control_mode

    def apply_deadzone(self, value):
        """针对Xbox摇杆死区处理"""
        if abs(value) < self.joystick_deadzone:
            return 0
        return value
    
    
controller=ArmController()