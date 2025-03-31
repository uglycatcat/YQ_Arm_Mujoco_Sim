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
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        glfw.set_key_callback(self.viewer.window, self.disable_mujoco_keys)
        # 初始化 Xbox 手柄
        self.init_xbox_controller()

        # 控制模式：0-键盘模式，1-Xbox 手柄模式
        self.control_mode = 0
        
        print("当前控制模式：键盘模式")
        

        # 手柄输入平滑滤波
        self.trans_filter = np.zeros(3)
        self.rot_filter = np.zeros(3)
        self.filter_alpha = 0.2  # 滤波系数


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
        trans = np.zeros(3)
        rot = np.zeros(3)

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
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                # 左摇杆控制 X 和 Y 轴平移
                if event.axis == self.AXIS_LEFT_X:
                    trans[1] = 1 if event.value > 0.5 else (-1 if event.value < -0.5 else 0)
                elif event.axis == self.AXIS_LEFT_Y:
                    trans[0] = -1 if event.value > 0.5 else (1 if event.value < -0.5 else 0)
                # 右摇杆控制 Z 轴平移
                elif event.axis == self.AXIS_RIGHT_Y:
                    trans[2] = -1 if event.value > 0.5 else (1 if event.value < -0.5 else 0)

            elif event.type == pygame.JOYBUTTONDOWN:
                # 检测按键按下
                if event.button in self.button_state:
                    self.button_state[event.button] = True

            elif event.type == pygame.JOYBUTTONUP:
                # 检测按键松开
                if event.button in self.button_state:
                    self.button_state[event.button] = False

        # 根据按键状态设置旋转
        if self.button_state[self.BUTTON_A]:  # A 按钮控制绕 X 轴正向旋转
            rot[0] = 1
        if self.button_state[self.BUTTON_B]:  # B 按钮控制绕 X 轴反向旋转
            rot[0] = -1
        if self.button_state[2]:  # X 按钮控制绕 Y 轴正向旋转
            rot[1] = 1
        if self.button_state[3]:  # Y 按钮控制绕 Y 轴反向旋转
            rot[1] = -1
        if self.button_state[4]:  # 左肩按钮控制绕 Z 轴正向旋转
            rot[2] = 1
        if self.button_state[5]:  # 右肩按钮控制绕 Z 轴反向旋转
            rot[2] = -1

        # 将输入信号映射到步长
        trans *= self.TRANS_STEP
        rot *= self.ROT_STEP

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
        while self.viewer.is_alive:
            has_input = False
            current_pos = self.data.xpos[self.end_effector_id].copy()
            current_rot = R.from_matrix(self.data.xmat[self.end_effector_id].reshape(3, 3))

            # 切换控制模式
            if keyboard.is_pressed('space'):
                self.control_mode = 1 - self.control_mode  # 切换模式
                print(f"切换到 {'Xbox 手柄' if self.control_mode else '键盘'} 模式")
                time.sleep(0.5)  # 防止连击

            # 处理输入
            if self.control_mode == 0:  # 键盘模式
                trans, rot = self.handle_keyboard_input()
            else:  # Xbox 手柄模式
                trans, rot = self.handle_xbox_input()

            if np.any(trans != 0):
                current_pos += trans
                has_input = True
            if np.any(rot != 0):
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
            if (time.time() - last_update) > 0.02:  # 50Hz
                self.viewer.render()
                last_update = time.time()

        # 程序结束时关闭串口
        protocol.stop()
        self.viewer.close()
        pygame.quit()

# 运行控制器
if __name__ == "__main__":
    model_dir = Path("urdf")
    model_path = str(model_dir / "scene.xml")
    controller = RobotArmController(model_path)
    controller.run()