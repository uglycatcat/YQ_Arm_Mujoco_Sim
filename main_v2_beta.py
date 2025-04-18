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

class RobotArmController:
    
    def __init__(self, model_path):
        # 加载 MuJoCo 模型文件并创建模型对象
        self.model = mj.MjModel.from_xml_path(model_path)
        # 创建与模型对应的数据对象，用于存储仿真状态
        self.data = mj.MjData(self.model)
        # 获取机械臂末端执行器（end effector）的 ID，用于后续控制和计算
        self.end_effector_id = self.model.body("link8").id
        # 获取最终位置mocap标记点的 ID，用于测试后续控制和计算
        self.mocap_body_id = self.model.body("target_marker").id
        self.mocap_index = self.mocap_body_id - (self.model.nbody - self.model.nmocap)
        # 获取joint6的mocap标记点ID，用于测试后续控制和计算
        self.joint6_mocap_id = self.model.body("joint6_marker").id
        self.joint6_mocap_index = self.joint6_mocap_id - (self.model.nbody - self.model.nmocap)
        # 执行一次前向动力学计算，初始化模型状态
        mj.mj_forward(self.model, self.data)
        # 打印帮助信息，显示控制器的使用说明
        self.help()
        # 机械臂可控关节索引
        self.control_list = [0, 1, 4, 7, 8, 9]
        # 运动参数
        self.TRANS_STEP = 0.01  # 平移步长 0.2cm（降低五倍）
        self.ROT_STEP = np.radians(1)  # 旋转步长 0.2度（降低五倍）
        # 初始化观察器
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, width=1200, height=800)
        glfw.set_key_callback(self.viewer.window, self.disable_mujoco_keys)
        
        print("当前键盘控制")
        
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
            - 机械臂末端旋转：
            O / K : 绕 X 轴旋转
            J / I : 绕 Y 轴旋转
            P / L : 绕 Z 轴旋转

            请确保 MuJoCo 界面处于激活状态，否则键盘输入可能无效。
            """)

    def run(self):
        """主循环"""
        last_update = time.time()
        print_counter = 0
        print_time_interval = 20  # 每20次打印一次间隔时间
        print_matrix_interval = 3000 # 每100次打印一次矩阵

        while self.viewer.is_alive if self.viewer else True:  # 根据是否创建窗口决定循环条件
            loop_start_time = time.time()
            
            # 执行前向动力学计算，更新模型状态
            has_input = False
            current_pos = self.data.xpos[self.end_effector_id].copy()
            current_rot = R.from_matrix(self.data.xmat[self.end_effector_id].reshape(3, 3))

            # 得到键盘输入
            trans, rot = self.handle_keyboard_input()

            # 修改输入检测逻辑,同时传递目标位置
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
                    
                    mj.mj_step(self.model, self.data)
                else:
                    print("IK求解失败")
                    
            # 控制更新频率
            if (time.time() - last_update) > 0.02:  # 50Hz
                self.viewer.render()
                last_update = time.time()

            # 输出控制循环耗时
            loop_end_time = time.time()
            print_counter += 1
            # 更新target_position目标位置显示
            self.data.mocap_pos[self.mocap_index] = current_pos
            # 更新target_position目标位置方向（四元数）
            quat_xyzw = current_rot.as_quat()  # 输出是 [x, y, z, w]
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]) # 转换为 [w, x, y, z]
            self.data.mocap_quat[self.mocap_index] = quat_wxyz
            
            # 更新joint6_position位置显示
            joint6_pos = self.data.xpos[self.model.body("link6").id].copy()
            self.data.mocap_pos[self.joint6_mocap_index] = joint6_pos
            # 更新joint6_position方向（四元数）
            joint6_rot = R.from_matrix(self.data.xmat[self.model.body("link5").id].reshape(3, 3))
            joint6_quat_xyzw = joint6_rot.as_quat()
            joint6_quat_wxyz = np.array([joint6_quat_xyzw[3], joint6_quat_xyzw[0], joint6_quat_xyzw[1], joint6_quat_xyzw[2]])
            self.data.mocap_quat[self.joint6_mocap_index] = joint6_quat_wxyz
            
            # 有输入时打印控制循环耗时
            if has_input and print_counter >= print_time_interval:
                print(f"控制循环耗时: {(loop_end_time - loop_start_time) * 1000:.2f}ms")       
                print_counter = 0
            # 无输入时打印当前位置和姿态
            if not has_input and print_counter >= print_matrix_interval:
                print(f"当前位置: {current_pos}")
                print(f"当前姿态: {current_rot.as_quat()}")      
                # print("mocap当前位置:", self.data.mocap_pos[self.mocap_index]) 
                print_counter = 0
            
        # 程序结束时关闭串口
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