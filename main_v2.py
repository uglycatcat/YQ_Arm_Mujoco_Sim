# 本文件由main.py文件修改而来
# 目的是优化其中的机械臂逆解算法
# 故本文件命名为V2版本

import mujoco as mj
import mujoco_viewer
import numpy as np
import keyboard
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time
from pathlib import Path
import glfw
import math
import pygame
#引入自定义串口协议
from protocol import protocol


class RobotArmController:
    
    def __init__(self, model_path):
        # 加载 MuJoCo 模型文件并创建模型对象
        self.model = mj.MjModel.from_xml_path(model_path)
        # 创建与模型对应的数据对象，用于存储仿真状态
        self.data = mj.MjData(self.model)
        # 获取机械臂末端执行器（end effector）的 ID，用于后续控制和计算
        self.end_effector_id = self.model.body("link6").id
        # 执行一次前向动力学计算，初始化模型状态
        mj.mj_forward(self.model, self.data)
        # 打印帮助信息，显示控制器的使用说明
        self.help()
        # 运动参数
        self.TRANS_STEP = 0.0005  # 平移步长 0.2cm（降低五倍）
        self.ROT_STEP = np.radians(0.05)  # 旋转步长 0.2度（降低五倍）
        # 机械臂可控关节索引
        self.control_list = [0, 1, 4, 7, 8, 9]
        # 启动串口通信协议
        protocol.start()
        # 初始化观察器
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, width=1200, height=800)
        glfw.set_key_callback(self.viewer.window, self.disable_mujoco_keys)
        
        print("当前键盘控制")
        
        # 添加手柄相关初始化
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.is_keyboard = True
        self.space_pressed = False
        
        # 手柄控制参数
        self.joystick_max_speed = 0.0005  # 最大移动速度
        self.joystick_filter_alpha = 0.2  # 平滑因子
        self.joystick_deadzone = 0.1      # 死区阈值
        
    def disable_mujoco_keys(self,window, key, scancode, action, mods):
        pass
    
    # 这里不执行任何操作，从而屏蔽默认快捷键
    def check_singularity(self):
        """检测奇异点"""
        # 计算雅可比矩阵
        jacobian_pos = np.zeros((3, self.model.nv))
        jacobian_rot = np.zeros((3, self.model.nv))
        mj.mj_jac(self.model, self.data, jacobian_pos, jacobian_rot, self.data.xpos[self.end_effector_id], self.end_effector_id)

        # 计算雅可比矩阵的秩
        jacobian = np.vstack((jacobian_pos, jacobian_rot))  # 6×n 矩阵
        rank = np.linalg.matrix_rank(jacobian)

        # 如果秩不足6，说明处于奇异点
        return rank < 6
    
    def solve_ik(self, target_pos):
        """逆运动学求解"""
        if self.check_singularity():
            print("警告：奇异点检测，放弃此次 IK 计算！")
            return None
        
        geometry_solution = self.solve_ik_geometry(target_pos)
        if geometry_solution is None:
            return
        # 更新前三个关节（0,1,4）
        self.data.qpos[0] = geometry_solution[0]
        self.data.qpos[1] = geometry_solution[1]
        self.data.qpos[4] = geometry_solution[2]
        
        self.data.qpos[2] = self.data.qpos[1]
        self.data.qpos[3] = self.data.qpos[1]
        self.data.qpos[5] = self.data.qpos[4]  # 根据实际机械结构调整
        self.data.qpos[6] = -self.data.qpos[5]
        
        # 保持最后三个关节为0
        self.data.qpos[7] = 0
        self.data.qpos[8] = 0
        self.data.qpos[9] = 0

        # 执行前向动力学计算
        mj.mj_forward(self.model, self.data)
        mj.mj_step(self.model, self.data)
        
        return geometry_solution

    def handle_joystick_input(self):
        """处理手柄输入"""
        if self.joystick is None:
            return np.zeros(3)
            
        pygame.event.pump()
        
        # 读取摇杆输入并应用死区
        left_y = self.apply_deadzone(self.joystick.get_axis(1))  # 左摇杆Y轴
        left_x = self.apply_deadzone(self.joystick.get_axis(0))  # 左摇杆X轴
        right_y = self.apply_deadzone(self.joystick.get_axis(3))  # 右摇杆Y轴
        
        # 检测手柄按键
        if self.joystick.get_button(0):  # A键
            protocol.change_mode(15)
        if self.joystick.get_button(1):  # B键
            protocol.change_mode(13)
        
        # 计算目标速度
        target_speeds = np.array([
            -left_y * self.joystick_max_speed,  # 注意Y轴方向
            left_x * self.joystick_max_speed,
            right_y * self.joystick_max_speed
        ])
        
        return target_speeds

    def apply_deadzone(self, value):
        """摇杆死区处理"""
        if abs(value) < self.joystick_deadzone:
            return 0
        return value

    def handle_keyboard_input(self):
        """处理键盘输入"""
        trans = np.zeros(3)

        # 处理空格键切换输入模式
        if keyboard.is_pressed('space'):
            if not self.space_pressed:
                self.space_pressed = True
                if self.is_keyboard and self.joystick is None:
                    print("无法切换到手柄模式，未检测到手柄。")
                else:
                    self.is_keyboard = not self.is_keyboard
                    print(f"切换到{'键盘' if self.is_keyboard else '手柄'}模式")
                    time.sleep(0.5)
        else:
            self.space_pressed = False

        # 根据当前模式处理输入
        if self.is_keyboard:
            # 原有的键盘控制逻辑
            if keyboard.is_pressed('s'): trans[0] += self.TRANS_STEP
            if keyboard.is_pressed('w'): trans[0] -= self.TRANS_STEP
            if keyboard.is_pressed('d'): trans[1] += self.TRANS_STEP
            if keyboard.is_pressed('a'): trans[1] -= self.TRANS_STEP
            if keyboard.is_pressed('UP'): trans[2] += self.TRANS_STEP
            if keyboard.is_pressed('DOWN'): trans[2] -= self.TRANS_STEP
        else:
            # 手柄控制
            trans = self.handle_joystick_input()

        return trans
    
    # 在main函数中运行的负责逆解的函数。传入目标位置三维坐标，返回三个关节的逆解弧度制角度
    def solve_ik_geometry(self,target_pos):
        """逆解几何计算(单位弧度制)"""
        
        # 将得到的位置传递给三维坐标
        target_3d_pos_x=target_pos[0]
        target_3d_pos_y=target_pos[1]
        target_3d_pos_z=target_pos[2]
        
        # 计算投影平面的法向量（Z轴和目标方向的叉积）
        z_axis = np.array([0, 0, 1])
        target_direction = np.array([target_3d_pos_x, target_3d_pos_y, target_3d_pos_z])
        target_direction = target_direction / np.linalg.norm(target_direction)
        plane_normal = np.cross(z_axis, target_direction)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        # 计算目标点在平面上的投影
        point = np.array([target_3d_pos_x, target_3d_pos_y, target_3d_pos_z])
        projection = point - np.dot(point, plane_normal) * plane_normal
        
        # 转换为二维坐标
        target_2d_pos_x = np.sqrt(projection[0]**2 + projection[1]**2)
        target_2d_pos_y = projection[2]
        
        # 处理X坐标符号（与原逻辑一致）
        if np.dot(projection[:2], target_direction[:2]) < 0:
            target_2d_pos_x = -target_2d_pos_x
        
        # 计算P4相对于P1点的偏移量
        x_offset = (target_2d_pos_x - 0.078304) - 0.03419422 - 0.0162178
        y_offset = (target_2d_pos_y - 0.0044872) - 0.05481078 - 0.136133
        
        # 计算P4P1的距离
        distance = math.sqrt(x_offset**2 + y_offset**2)
        
        # P1P2和P3P4两段长度固定
        arm_length_1=0.4000
        arm_length_2=0.4000
        
        # 三边构成一个三角形
        if distance>arm_length_1+arm_length_2:
            print("警告：目标点超出机械臂可到达范围！")
            return None
        # 计算theta_2（arm_length_1和distance构成的角）
        theta_2 = math.acos((arm_length_1**2 + distance**2 - arm_length_2**2) / (2 * arm_length_1 * distance))
        # 计算theta_3（arm_length_1和arm_length_2构成的角）
        theta_3 = math.acos((arm_length_1**2 + arm_length_2**2 - distance**2) / (2 * arm_length_1 * arm_length_2))
        
        # temp1得到P4P1两点的y/x的正切值对应的弧度制
        temp1 = math.atan2(y_offset, x_offset)
        
        # 根据计算结果得到真正的theta_2和theta_3
        theta_1 = -math.atan2(target_3d_pos_y, -target_3d_pos_x)
        theta_2 = temp1+theta_2-math.radians(120)
        theta_3 = (math.pi/2-(math.pi-theta_2-math.radians(120))+theta_3)-math.radians(70)
        
        target_theta=[theta_1, -theta_2, theta_3]
        
        # 返回弧度制的计算结果
        return target_theta
    
    def help(self):
        print("""
            =============================
            机械臂控制器 使用说明
            =============================

            [ 键盘模式 控制按键 ]
            - 机械臂末端平移：
            W / S : 前进 / 后退
            A / D : 左移 / 右移
            ↑ / ↓ : 上移 / 下移

            请确保 MuJoCo 界面处于激活状态，否则键盘输入可能无效。
            """)

    def run(self):
        """主循环"""
        last_update = time.time()
        print_counter = 0
        
        while self.viewer.is_alive if self.viewer else True:
            # 检查手柄连接状态
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

            loop_start_time = time.time()
            
            # 获取当前末端执行器的位置和姿态
            current_pos = self.data.xpos[self.end_effector_id].copy()
            
            # 得到键盘输入
            trans= self.handle_keyboard_input()

            # 修改输入检测逻辑,同时传递目标位置
            if np.any(np.abs(trans) > 1e-5):
                current_pos += trans
            
            self.solve_ik(current_pos)
            # 更新关节角度到串口协议
            protocol.update_angles([self.data.qpos[i] for i in self.control_list])
            
            # 控制更新频率
            if (time.time() - last_update) > 0.02:  # 50Hz
                self.viewer.render()
                last_update = time.time()

            # 输出控制循环耗时
            loop_end_time = time.time()
            print_counter += 1
            
            # 打印解偏差
            if print_counter>=2000:
                qpos_0 = f"{math.degrees(self.data.qpos[0]):.6f}"
                qpos_1 = f"{math.degrees(self.data.qpos[1]):.6f}"
                qpos_4 = f"{math.degrees(self.data.qpos[4]):.6f}"
                print(f"当前位置：{current_pos}")
                print(f"仿真角度：[{qpos_0}, {qpos_1}, {qpos_4}]")
                print(f"控制循环耗时: {(loop_end_time - loop_start_time) * 1000:.2f}ms")
                print_counter = 0
            
        # 程序结束时关闭串口
        protocol.stop()
        # 程序结束时关闭窗口
        if self.viewer:
            self.viewer.close()

# 运行控制器
if __name__ == "__main__":
    # 定义模型目录，存放 URDF 文件
    model_dir = Path("urdf")
    # 构建模型文件的完整路径，指向 scene.xml 文件
    model_path = str(model_dir / "scene.xml")
    # 创建 RobotArmController 实例，传入模型路径
    controller = RobotArmController(model_path)
    # 调用控制器的 run 方法，开始主循环
    controller.run()
    
    
class InputController:
    pre_angles = [0.0, 0.0, 0.0]  # 存储前三轴角度值
    after_angles = [0.0, 0.0, 0.0] # 存储后三轴角度值
    is_keyboard = True # 判断是否使用键盘输入
    space_pressed = False # 判断是否按下空格键
    joystick = None  # 全局手柄对象，只初始化一次

    # 确保角度在0到2*pi之间
    def normalize_angle(self,angle):
        return angle % (2 * np.pi)

    # 获取输入的角度值
    # 获取device_input_angles()的长度为3的数组
    # 获取cv_input_angles()的长度为3的数组
    # 返回拼接后数组
    def get_input_angless(self):
        """获取输入的角度值"""
        return self.device_input_angles() + self.cv_input_angles()  # 拼接输入角度

    # 需要先检验是否成功连接键盘或者Xbox
    # 获取通过键盘或者Xbox输入的角度值
    # 并且根据键盘每按下"space"键，切换设备
    # 返回长度为3的数组
    def device_input_angles(self):
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
            pre_angles = self.keyboard_input()
        else:
            pre_angles = self.xbox_input()
        return pre_angles

    # 检测 Xbox 手柄是否连接
    def is_xbox_connected(self):
        return pygame.joystick.get_count() > 0

    # 获取通过CV解算出的角度值
    # 返回长度为3的数组
    # 每个轴的数据范围为0-2*pi四字节浮点数
    # 初始值为0
    def cv_input_angles(self):
        after_angles = [0.0, 0.0, 0.0]
        # 这里可以添加CV解算的逻辑
        return after_angles  # 返回固定值作为占位

    # 获取通过键盘输入的角度值
    # 返回长度为3的数组
    # 其中"a""d"控制第一轴，"w""s"控制第二轴，"q""e"控制第三轴
    # 每个轴的数据范围为0-2*pi四字节浮点数
    # 初始值为0
    def keyboard_input(self):
        global pre_angles  # 使用全局变量
        
        # 初始化参数
        if not hasattr(self.keyboard_input, 'max_speed'):
            self.keyboard_input.max_speed = 0.15
        if not hasattr(self.xbox_input, 'smooth_factor'):
            self.keyboard_input.smooth_factor = 0.002 #数值越高 越快抵达目标速度
        if not hasattr(self.xbox_input, 'current_speeds'):
            self.keyboard_input.current_speeds = [0.0, 0.0, 0.0]  # 每个轴的当前速度
        
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
                target_speed = direction * self.keyboard_input.max_speed
                # 平滑过渡到目标速度
                self.keyboard_input.current_speeds[axis] += self.keyboard_input.smooth_factor * (target_speed - keyboard_input.current_speeds[axis])
        
        # 如果没有按键被按下，逐渐减速
        if not any_key_pressed:
            for i in range(3):
                if abs(self.keyboard_input.current_speeds[i]) > 0.0001:  # 小阈值防止抖动
                    self.keyboard_input.current_speeds[i] *= (1 - self.keyboard_input.smooth_factor)
                else:
                    self.keyboard_input.current_speeds[i] = 0.0
        
        # 应用速度更新角度（确保速度不超过最大速度）
        for i in range(3):
            # 限制速度范围（正负方向）
            self.keyboard_input.current_speeds[i] = max(-self.keyboard_input.max_speed, 
                                                min(self.keyboard_input.max_speed, 
                                                    self.keyboard_input.current_speeds[i]))
            pre_angles[i] = self.normalize_angle(pre_angles[i] + self.keyboard_input.current_speeds[i])
        
        # 控制最大速度
        if keyboard.is_pressed('z') and self.keyboard_input.max_speed < 0.3:
            self.keyboard_input.max_speed = min(self.keyboard_input.max_speed + 0.05, 0.3)
            time.sleep(0.1)
        if keyboard.is_pressed('x') and self.keyboard_input.max_speed > 0.05:
            self.keyboard_input.max_speed = max(self.keyboard_input.max_speed - 0.05, 0.05)
            time.sleep(0.1)
        
        return pre_angles

    # 获取通过Xbox输入的角度值
    # 返回长度为3的数组
    # 其中左摇杆横轴表示第一轴，左摇杆纵轴表示第二轴，右摇杆纵轴表示第三轴
    # 每个轴的数据范围为0-2*pi四字节浮点数
    # 初始值为0
    def xbox_input(self):
        global pre_angles, joystick

        pygame.event.pump()

        # 初始化配置参数（只运行一次）
        if not hasattr(self.xbox_input, 'max_speed',):
            self.xbox_input.max_speed = 0.001  # 可调最大角速度（单位：弧度/帧）
        if not hasattr(self.xbox_input, 'filter_alpha'):
            self.xbox_input.filter_alpha = 0.2  # 平滑因子 (0-1)，越小越平滑

        # 读取摇杆输入，范围 -1 到 1
        left_x = self.apply_deadzone(joystick.get_axis(0))
        left_y = self.apply_deadzone(joystick.get_axis(1))
        right_y = self.apply_deadzone(joystick.get_axis(3))

        # 把摇杆值乘以最大角速度，得到目标速度
        target_speeds = [
            left_x * self.xbox_input.max_speed,
            left_y * self.xbox_input.max_speed,
            right_y * self.xbox_input.max_speed
        ]

        # 平滑：一阶低通滤波，或者说指数滑动平均方式更新角度
        for i in range(3):
            delta_angle = target_speeds[i]
            pre_angles[i] = self.normalize_angle(pre_angles[i] + delta_angle * xbox_input.filter_alpha)

        return pre_angles

    def apply_deadzone(self,value, deadzone=0.1):
        """摇杆死区处理"""
        if abs(value) < deadzone:
            return 0
        return value

    def __init__(self):
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

                input_angles = self.get_input_angles()
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