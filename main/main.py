# 目的是优化其中的机械臂逆解算法
# 本文件使用几何解逆解将得到的结果传给机械臂
# 同时使用Xbox或者键盘接收信息
# 通过串口协议将信息传递给下位机
# 本文件是相对于main.py的几何解优化版本
# 计算复杂度由O（n*k）降低为O（1）
# sudo /home/sunrise/miniconda3/envs/mujoco_env/bin/python main/main.py

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
from Protocol import protocol


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
        self.TRANS_STEP = 0.0001  # 平移步长 0.2cm（降低五倍）
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
        self.joystick_max_speed = 0.0001  # 最大移动速度
        self.joystick_filter_alpha = 0.01  # 平滑因子
        self.joystick_deadzone = 0.001      # 死区阈值
        
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
        left_y = -self.apply_deadzone(self.joystick.get_axis(1))  # 左摇杆Y轴
        left_x = self.apply_deadzone(self.joystick.get_axis(0))  # 左摇杆X轴
        right_y = -self.apply_deadzone(self.joystick.get_axis(3))  # 右摇杆Y轴
        
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
        
        target_theta=[theta_1+0.00005, -theta_2, theta_3]
        
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