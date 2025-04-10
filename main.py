# if you want to run this project on linux system
# you need to run this command interminal
# sudo /home/sunrise/miniconda3/envs/mujoco_env/bin/python main.py
# because linux dont support normal user receive command from keyboard
# so the keyboard library will be error

import mujoco as mj
import mujoco_viewer
import numpy as np
import pygame
import keyboard
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time
from pathlib import Path
import glfw

#引入自定义串口协议
from protocol import protocol

# 在文件开头定义全局变量
if_mujoco_render = 0  
# 默认为1，表示创建窗口
# 更改为0，表示不创建窗口

class RobotArmController:
    def __init__(self, model_path):
        # 初始化 MuJoCo 模型和数据
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        self.end_effector_id = self.model.body("link8").id
        mj.mj_forward(self.model, self.data)
        self.help()

        # 启动串口通信协议
        protocol.start()

        # 机械臂可控关节索引
        self.control_list = [0, 1, 4, 7, 8, 9]

        # 运动参数
        self.TRANS_STEP = 0.01  # 平移步长 0.2cm（降低五倍）
        self.ROT_STEP = np.radians(1)  # 旋转步长 0.2度（降低五倍）

        # 初始化观察器
        if if_mujoco_render:  # 根据全局变量决定是否创建窗口
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, width=800, height=500)
            glfw.set_key_callback(self.viewer.window, self.disable_mujoco_keys)
        else:
            self.viewer = None  # 不创建窗口

        # 初始化 Xbox 手柄
        self.init_xbox_controller()

        # 控制模式：0-键盘模式，1-Xbox 手柄模式
        self.control_mode = 0
        
        print("当前控制模式：键盘模式")

        # 手柄输入平滑滤波
        self.current_trans = np.zeros(3)  # 当前实际输出的平移值
        self.current_rot = np.zeros(3)    # 当前实际输出的旋转值
        self.smooth_factor = 0.3          # 平滑系数

        # 关节控制模式的平滑滤波
        self.current_joint_values = np.zeros(3)  # 当前实际输出的关节值
        self.smooth_factor_joint = 0.1          # 关节控制的平滑系数
        
    def disable_mujoco_keys(self,window, key, scancode, action, mods):
        pass
       # 这里不执行任何操作，从而屏蔽默认快捷键

    def init_xbox_controller(self):
        """初始化 Xbox 手柄"""
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("未检测到手柄！请连接 Xbox 手柄后重试。")
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"已连接手柄：{self.joystick.get_name()}")
        
        # 手柄输入映射
        self.AXIS_LEFT_X = 0  # 左摇杆 X 轴
        self.AXIS_LEFT_Y = 1  # 左摇杆 Y 轴
        self.AXIS_RIGHT_X = 3  # 右摇杆 X 轴
        self.AXIS_RIGHT_Y = 2  # 右摇杆 Y 轴
        self.BUTTON_A = 0  # A 按钮
        self.BUTTON_B = 1  # B 按钮

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

    def solve_ik(self, target_pos, target_rot):
        """逆运动学求解"""
        if self.check_singularity():
            print("警告：奇异点检测，放弃此次 IK 计算！")
            return None

        def objective(q):
            # 仅更新可控关节
            for i, joint_idx in enumerate(self.control_list):
                self.data.qpos[joint_idx] = q[i]
            
            self.data.qpos[2] = self.data.qpos[3] = self.data.qpos[1]  # joint3 = joint2_2
            self.data.qpos[5] = self.data.qpos[4]  # 根据实际机械结构调整
            self.data.qpos[6] = -self.data.qpos[5]
            mj.mj_forward(self.model, self.data)
            
            # 计算误差
            pos_err = np.linalg.norm(self.data.xpos[self.end_effector_id] - target_pos)
            orient_err = np.linalg.norm(self.data.xmat[self.end_effector_id].reshape(3, 3) - target_rot.as_matrix(), ord='fro') / 3
            return pos_err + 0.5 * orient_err  # 调整权重平衡
        
        # 关节角度约束
        constraints = [
            {"type": "ineq", "fun": lambda q, i=i: q[i] - self.model.jnt_range[self.control_list[i], 0]} for i in range(len(self.control_list))
        ] + [
            {"type": "ineq", "fun": lambda q, i=i: self.model.jnt_range[self.control_list[i], 1] - q[i]} for i in range(len(self.control_list))
        ]
        
        # 初始值
        q_init = [self.data.qpos[i] for i in self.control_list]
        res = minimize(objective, q_init, method='SLSQP', constraints=constraints)
        return res

    def handle_keyboard_input(self):
        """处理键盘输入"""
        trans = np.zeros(3)
        rot = np.zeros(3)

        # 处理平移输入
        if keyboard.is_pressed('s'): trans[0] += self.TRANS_STEP
        if keyboard.is_pressed('w'): trans[0] -= self.TRANS_STEP
        if keyboard.is_pressed('d'): trans[1] += self.TRANS_STEP
        if keyboard.is_pressed('a'): trans[1] -= self.TRANS_STEP
        if keyboard.is_pressed('UP'): trans[2] += self.TRANS_STEP
        if keyboard.is_pressed('DOWN'): trans[2] -= self.TRANS_STEP

        # 处理旋转输入
        if keyboard.is_pressed('o'): rot[0] += self.ROT_STEP
        if keyboard.is_pressed('k'): rot[0] -= self.ROT_STEP
        if keyboard.is_pressed('j'): rot[1] += self.ROT_STEP
        if keyboard.is_pressed('i'): rot[1] -= self.ROT_STEP
        if keyboard.is_pressed('p'): rot[2] += self.ROT_STEP
        if keyboard.is_pressed('l'): rot[2] -= self.ROT_STEP

        return trans, rot

    def handle_xbox_input(self):
        """处理 Xbox 手柄输入"""
        target_trans = np.zeros(3)
        target_rot = np.zeros(3)

        # 初始化按键状态（如果尚未初始化）
        if not hasattr(self, 'button_state'):
            self.button_state = {
                self.BUTTON_A: False,  # A 按钮
                self.BUTTON_B: False,  # B 按钮
                2: False,  # X 按钮
                3: False,  # Y 按钮
                4: False,  # 左肩按钮
                5: False,  # 右肩按钮
            }

        # 处理事件
        pygame.event.pump()

        # 读取摇杆输入作为目标值
        left_x = self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_X))
        left_y = self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_Y))
        right_y = self.apply_deadzone(self.joystick.get_axis(self.AXIS_RIGHT_Y))

        # 映射摇杆输入到目标平移值
        if abs(left_x) > 0.1:
            target_trans[1] = (abs(left_x) - 0.1) * (0.9 / 0.9) * (1 if left_x > 0 else -1)
        if abs(left_y) > 0.1:
            target_trans[0] = (abs(left_y) - 0.1) * (0.9 / 0.9) * (1 if left_y > 0 else -1)
        if abs(right_y) > 0.1:
            target_trans[2] = (abs(right_y) - 0.1) * (0.9 / 0.9) * (-1 if right_y > 0 else 1)

        # 处理按钮事件
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button in self.button_state:
                    self.button_state[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in self.button_state:
                    self.button_state[event.button] = False

        # 根据按键状态设置目标旋转值
        if self.button_state[self.BUTTON_A]: target_rot[0] = 1
        if self.button_state[self.BUTTON_B]: target_rot[0] = -1
        if self.button_state[2]: target_rot[1] = 1
        if self.button_state[3]: target_rot[1] = -1
        if self.button_state[4]: target_rot[2] = 1
        if self.button_state[5]: target_rot[2] = -1

        # 平滑过渡到目标值
        self.current_trans += self.smooth_factor * (target_trans - self.current_trans)
        self.current_rot += self.smooth_factor * (target_rot - self.current_rot)

        # 将平滑后的输入信号映射到步长
        trans = self.current_trans * self.TRANS_STEP
        rot = self.current_rot * self.ROT_STEP

        return trans, rot

    def apply_deadzone(self, value, deadzone=0.1):
        """摇杆死区处理"""
        if abs(value) < deadzone:
            return 0
        return value

    def help(self):
        print("""
            =============================
            机械臂控制器 使用说明
            =============================

            [ 控制模式切换 ]
            - 按下 空格键 切换控制模式：
            * 键盘模式
            * Xbox 手柄模式

            [ 键盘模式 控制按键 ]
            - 机械臂末端平移：
            W / S : 前进 / 后退
            A / D : 左移 / 右移
            ↑ / ↓ : 上移 / 下移
            - 机械臂末端旋转：
            O / K : 绕 X 轴旋转
            J / I : 绕 Y 轴旋转
            P / L : 绕 Z 轴旋转

            [ 手柄模式 控制方式 ]
            - 左摇杆 控制 X/Y 平移
            - 右摇杆 控制 Z 轴平移
            - A / B 按钮 控制绕 X 轴旋转
            - X / Y 按钮 控制绕 Y 轴旋转
            - 左 / 右肩键 控制绕 Z 轴旋转

            请确保 MuJoCo 界面处于激活状态，否则键盘输入可能无效。
            """)

    def run(self):
        """主循环"""
        last_update = time.time()
        print_counter = 0
        print_interval = 10  # 每10次打印一次

        while self.viewer.is_alive if self.viewer else True:  # 根据是否创建窗口决定循环条件
            loop_start_time = time.time()
            
            has_input = False
            current_pos = self.data.xpos[self.end_effector_id].copy()
            current_rot = R.from_matrix(self.data.xmat[self.end_effector_id].reshape(3, 3))

            # 切换控制模式
            if keyboard.is_pressed('space'):
                self.control_mode = (self.control_mode + 1) % 3  # 在三种模式间切换
                mode_names = ['键盘', 'Xbox 手柄', 'Xbox 手柄关节']
                print(f"切换到 {mode_names[self.control_mode]} 模式")
                time.sleep(0.5)  # 防止连击

            # 处理输入
            if self.control_mode == 2:
                #该模式下，使用Xbox手柄控制关节角度
                trans, rot = self.handle_xbox_input()
                
                # 设置关节角度变化步长
                joint_step = 0.01
                
                # 目标关节值
                target_joint_values = trans[:3]  # 只取前三个关节的控制量
                
                # 平滑过渡到目标值
                self.current_joint_values += self.smooth_factor_joint * (target_joint_values - self.current_joint_values)
                
                # 更新所有控制关节的角度
                for i, joint_idx in enumerate(self.control_list[:3]):
                    # 使用平滑后的值计算新角度
                    new_angle = self.data.qpos[joint_idx] + self.current_joint_values[i] * joint_step
                    new_angle = np.clip(new_angle, 
                                      self.model.jnt_range[joint_idx, 0],
                                      self.model.jnt_range[joint_idx, 1])
                    
                    self.data.qpos[joint_idx] = new_angle
                    self.data.ctrl[joint_idx] = new_angle
                
                mj.mj_forward(self.model, self.data)
                
                # 更新关节角度到串口协议
                protocol.update_angles([self.data.qpos[i] for i in self.control_list])
                    
            else:
                # 末端控制模式，需要求解IK
                if self.control_mode == 0:  # 键盘模式
                    trans, rot = self.handle_keyboard_input()
                else:  # Xbox 手柄模式
                    trans, rot = self.handle_xbox_input()

                # 修改输入检测逻辑
                if np.any(np.abs(trans) > 1e-6) or np.any(np.abs(rot) > 1e-6):
                    current_pos += trans
                    current_rot = R.from_rotvec(rot) * current_rot
                    has_input = True

                # 仅在有输入时执行IK
                if has_input:
                    res = self.solve_ik(current_pos, current_rot)
                    if res is not None and res.success:
                        for i, joint_idx in enumerate(self.control_list):
                            self.data.qpos[joint_idx] = res.x[i]
                        # 更新关节角度到串口协议
                        protocol.update_angles([self.data.qpos[i] for i in self.control_list])
                        mj.mj_step(self.model, self.data)
                    else:
                        print("IK求解失败")

            # 控制更新频率
            if if_mujoco_render and (time.time() - last_update) > 0.02:  # 50Hz
                self.viewer.render()
                last_update = time.time()

            # 输出控制循环耗时
            loop_end_time = time.time()
            print_counter += 1
            if print_counter >= print_interval:
                print(f"控制循环耗时: {(loop_end_time - loop_start_time) * 1000:.2f}ms")
                print_counter = 0

        # 程序结束时关闭串口
        protocol.stop()
        if self.viewer:
            self.viewer.close()
        pygame.quit()

# 运行控制器
if __name__ == "__main__":
    model_dir = Path("urdf")
    model_path = str(model_dir / "scene.xml")
    controller = RobotArmController(model_path)
    controller.run()