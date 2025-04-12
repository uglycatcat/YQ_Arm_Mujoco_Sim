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
rotation_speed = 0.001 # 控制旋转速度

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
    
    # 检测空格键按下
    if keyboard.is_pressed('space'):
        if not space_pressed:  # 只在第一次按下时触发
            space_pressed = True
            if not is_xbox_connected():
                print("无法切换到手柄模式，手柄未连接。")
            else:
                is_keyboard = not is_keyboard
                if is_keyboard:
                    print("切换到键盘模式")
                else:
                    print("切换到手柄模式")
                time.sleep(0.5)
    else:
        space_pressed = False  # 重置标志位
    
    if is_keyboard:
        pre_angles = keyboard_input()
    else:
        pre_angles = xbox_input()
    return pre_angles

# 检测 Xbox 手柄是否连接
def is_xbox_connected():
    pygame.init()
    pygame.joystick.init()
    connected = pygame.joystick.get_count() > 0
    if connected:
        print("已连接手柄。")
    else:
        print("未检测到手柄，请连接后重试。")
    return connected

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
    global pre_angles,rotation_speed  # 使用全局变量
    if keyboard.is_pressed('a'):
        pre_angles[0] = normalize_angle(pre_angles[0] + rotation_speed)  # 控制第一轴
    if keyboard.is_pressed('d'):
        pre_angles[0] = normalize_angle(pre_angles[0] - rotation_speed)
    if keyboard.is_pressed('w'):
        pre_angles[1] = normalize_angle(pre_angles[1] + rotation_speed)  # 控制第二轴
    if keyboard.is_pressed('s'):
        pre_angles[1] = normalize_angle(pre_angles[1] - rotation_speed)
    if keyboard.is_pressed('q'):
        pre_angles[2] = normalize_angle(pre_angles[2] + rotation_speed)  # 控制第三轴
    if keyboard.is_pressed('e'):
        pre_angles[2] = normalize_angle(pre_angles[2] - rotation_speed)
    return pre_angles

# 获取通过Xbox输入的角度值
# 返回长度为3的数组
# 其中左摇杆横轴表示第一轴，左摇杆纵轴表示第二轴，右摇杆纵轴表示第三轴
# 每个轴的数据范围为0-2*pi四字节浮点数
# 初始值为0
def xbox_input():
    global pre_angles  # 使用全局变量
    pygame.event.pump()  # 处理事件

    joystick = pygame.joystick.Joystick(0)  # 获取手柄对象
    joystick.init()  # 初始化手柄

    # 读取摇杆输入作为目标值
    left_x = apply_deadzone(joystick.get_axis(0))  # 左摇杆 X 轴
    left_y = apply_deadzone(joystick.get_axis(1))  # 左摇杆 Y 轴
    right_y = apply_deadzone(joystick.get_axis(2))  # 右摇杆 Y 轴

    # 映射摇杆输入到目标角度值
    pre_angles[0] = normalize_angle(pre_angles[0] + left_x * rotation_speed)  # 乘以一个系数以调整灵敏度
    pre_angles[1] = normalize_angle(pre_angles[1] + left_y * rotation_speed)
    pre_angles[2] = normalize_angle(pre_angles[2] + right_y * rotation_speed)

    return pre_angles  # 返回更新后的角度值

def apply_deadzone(value, deadzone=0.1):
    """摇杆死区处理"""
    if abs(value) < deadzone:
        return 0
    return value

def main():
    """主程序入口"""
    pygame.init()
    clock = pygame.time.Clock()

    # 启动协议
    protocol.start()

    count = 0  # 初始化计数器

    try:
        while True:
            # 获取输入角度
            input_angles = get_input_angles()
            # 更新协议中的关节角度
            protocol.update_angles(input_angles)

            # 每1000次循环在终端打印一次数据
            if count % 1000 == 0:
                print(input_angles)
            count += 1

            # 发送数据
            time.sleep(0.001)  # 1000Hz
            clock.tick(1000)  # 限制帧率
    except KeyboardInterrupt:
        print("程序中断，正在停止...")
    finally:
        protocol.stop()
        pygame.quit()

if __name__ == "__main__":
    is_xbox_connected()  # 检测手柄连接状态
    main()


