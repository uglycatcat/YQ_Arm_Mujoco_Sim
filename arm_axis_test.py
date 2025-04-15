# 该文件为单独程序，专用于进行单轴特殊测试
# 该文件执行后调用protocol内协议内容，向下位机发送六轴位置信息，但不经过位置解算等任何特殊处理
# 其中，前三轴的数据从键盘或者Xbox蓝牙连接后传入；
# 后三轴数据由CV部分解算得出传递给本程序并发送给下位机(暂时使用固定值，只需留出函数接口即可)
# 本程序内包含两进程，一个用于处理键盘活着Xbox的输入，一个用于处理下位机协议发送（每次发送六个位置数据）
# 协议格式，数据内容参考protocol.py，发送频率为50Hz
# 当前程序有bug，无论window还是linux，都无法接收蓝牙Xbox数据
# 由于linux操作系统对键盘的权限管理，运行此程序需指令
# sudo /home/sunrise/miniconda3/envs/mujoco_env/bin/python arm_axis_test.py
# Linux系统内报错ALSA相关为音频流报错，无影响（不知原因）

import mujoco_viewer
import pygame
import keyboard
import time
from protocol import protocol
import numpy as np

# 初始化全局变量
pre_angles = [0.0, 0.0, 0.0]  # 存储前三轴角度值
after_angles = [0.0, 0.0, 0.0] # 存储后三轴角度值
is_keyboard = True # 判断是否使用键盘输入
space_pressed = False # 判断是否按下空格键
joystick = None  # 全局手柄对象，只初始化一次

# 确保角度在0到2*pi之间
def normalize_angle(angle):
    return angle % (2 * np.pi)

# 获取输入的角度值
# 获取device_input_angles()的长度为3的数组
# 获取cv_input_angles()的长度为3的数组
# 返回拼接后数组
def get_input_angles():
    """获取输入的角度值"""
    return device_input_angles() + cv_input_angles()  # 拼接输入角度

# 需要先检验是否成功连接键盘或者Xbox
# 获取通过键盘或者Xbox输入的角度值
# 并且根据键盘每按下"space"键，切换设备
# 返回长度为3的数组
def device_input_angles():
    global pre_angles, is_keyboard, space_pressed
    
    # 检测手柄连接状态
    if keyboard.is_pressed('space'):
        if not space_pressed:
            space_pressed = True
            # 确保只有在手柄可用时才允许切换到手柄
            if is_keyboard and joystick is None:
                print("无法切换到手柄模式，未检测到手柄。")
            else:
                is_keyboard = not is_keyboard
                print("切换到{}模式".format("键盘" if is_keyboard else "手柄"))
                time.sleep(0.5)
    else:
        space_pressed = False
    
    if is_keyboard:
        pre_angles = keyboard_input()
    else:
        pre_angles = xbox_input()
    return pre_angles

# 检测 Xbox 手柄是否连接
def is_xbox_connected():
    return pygame.joystick.get_count() > 0

# 获取通过CV解算出的角度值
# 返回长度为3的数组
# 每个轴的数据范围为0-2*pi四字节浮点数
# 初始值为0
def cv_input_angles():
    after_angles = [0.0, 0.0, 0.0]
    # 这里可以添加CV解算的逻辑
    return after_angles  # 返回固定值作为占位

# 获取通过键盘输入的角度值
# 返回长度为3的数组
# 其中"a""d"控制第一轴，"w""s"控制第二轴，"q""e"控制第三轴
# 每个轴的数据范围为0-2*pi四字节浮点数
# 初始值为0
def keyboard_input():
    global pre_angles  # 使用全局变量
    
    # 初始化参数
    if not hasattr(keyboard_input, 'max_speed'):
        keyboard_input.max_speed = 0.0008
        keyboard_input.smooth_factor = 0.01  # 平滑因子，值越大越平滑
        keyboard_input.current_speeds = [0.0, 0.0, 0.0]  # 每个轴的当前速度
    
    # 定义按键与轴和方向的映射
    key_actions = {
        'a': (0, 1),    # 轴0，正向
        'd': (0, -1),   # 轴0，负向
        'w': (1, 1),    # 轴1，正向
        's': (1, -1),   # 轴1，负向
        'q': (2, 1),    # 轴2，正向
        'e': (2, -1)    # 轴2，负向
    }
    
    # 检查每个按键并更新速度
    any_key_pressed = False
    for key, (axis, direction) in key_actions.items():
        if keyboard.is_pressed(key):
            any_key_pressed = True
            # 计算目标速度（考虑方向）
            target_speed = direction * keyboard_input.max_speed
            # 平滑过渡到目标速度
            keyboard_input.current_speeds[axis] += keyboard_input.smooth_factor * (target_speed - keyboard_input.current_speeds[axis])
    
    # 如果没有按键被按下，逐渐减速
    if not any_key_pressed:
        for i in range(3):
            if abs(keyboard_input.current_speeds[i]) > 0.00001:  # 小阈值防止抖动
                keyboard_input.current_speeds[i] *= (1 - keyboard_input.smooth_factor)
            else:
                keyboard_input.current_speeds[i] = 0.0
    
    # 应用速度更新角度（确保速度不超过最大速度）
    for i in range(3):
        # 限制速度范围（正负方向）
        keyboard_input.current_speeds[i] = max(-keyboard_input.max_speed, 
                                            min(keyboard_input.max_speed, 
                                                keyboard_input.current_speeds[i]))
        pre_angles[i] = normalize_angle(pre_angles[i] + keyboard_input.current_speeds[i])
    
    # 控制最大速度
    if keyboard.is_pressed('z') and keyboard_input.max_speed < 0.0016:
        keyboard_input.max_speed = min(keyboard_input.max_speed + 0.0002, 0.0016)
        time.sleep(0.1)
    if keyboard.is_pressed('x') and keyboard_input.max_speed > 0.0004:
        keyboard_input.max_speed = max(keyboard_input.max_speed - 0.0002, 0.0004)
        time.sleep(0.1)
    
    return pre_angles

# 获取通过Xbox输入的角度值
# 返回长度为3的数组
# 其中左摇杆横轴表示第一轴，左摇杆纵轴表示第二轴，右摇杆纵轴表示第三轴
# 每个轴的数据范围为0-2*pi四字节浮点数
# 初始值为0
def xbox_input():
    global pre_angles, joystick

    pygame.event.pump()

    # 检查是否有 filter_factor 属性
    if not hasattr(xbox_input, 'filter_factor'):
        xbox_input.filter_factor = 0.001 

    # 获取摇杆值
    left_x = apply_deadzone(joystick.get_axis(0))  # 左摇杆X
    left_y = apply_deadzone(joystick.get_axis(1))  # 左摇杆Y
    right_y = apply_deadzone(joystick.get_axis(3))  # 右摇杆Y（通常是axis 3）

    # 更新角度
    pre_angles[0] = normalize_angle(pre_angles[0] + left_x * xbox_input.filter_factor)
    pre_angles[1] = normalize_angle(pre_angles[1] + left_y * xbox_input.filter_factor)
    pre_angles[2] = normalize_angle(pre_angles[2] + right_y * xbox_input.filter_factor)

    return pre_angles

def apply_deadzone(value, deadzone=0.1):
    """摇杆死区处理"""
    if abs(value) < deadzone:
        return 0
    return value

def main():
    global joystick, is_keyboard
    pygame.init()
    pygame.joystick.init()

    clock = pygame.time.Clock()
    protocol.start()

    count = 0

    try:
        while True:
            # 检测手柄连接状态是否变化
            current_joystick_count = pygame.joystick.get_count()
            if current_joystick_count > 0:
                if joystick is None:
                    # 如果之前未连接，现在连接了，就初始化
                    joystick = pygame.joystick.Joystick(0)
                    joystick.init()
                    print(f"手柄已连接：{joystick.get_name()}")
            else:
                if joystick is not None:
                    # 如果之前有连接，现在断开了，就清空
                    print("手柄已断开，自动切换到键盘模式")
                    joystick = None
                    is_keyboard = True

            input_angles = get_input_angles()
            protocol.update_angles(input_angles)

            if count % 1000 == 0:
                print(input_angles)
            count += 1

            # 1000Hz
            time.sleep(0.001)
            clock.tick(1000)
    except KeyboardInterrupt:
        print("程序中断，正在停止...")
    finally:
        protocol.stop()
        pygame.quit()

if __name__ == "__main__":
    main()
