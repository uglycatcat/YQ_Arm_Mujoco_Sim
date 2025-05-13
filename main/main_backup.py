import numpy as np
import mujoco as mj
import mujoco_viewer
import glfw
import time
import math
from pathlib import Path
from SolveIK import solveik
from Controller import controller
from DebugGUI import debuggui

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
        # 机械臂可控关节索引
        self.control_list = [0, 1, 4, 7, 8, 9]
        # 初始化观察器
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, width=1200, height=800)
        glfw.set_key_callback(self.viewer.window, self.disable_mujoco_keys)
        
        controller.start()
        debuggui.start()
        
    def disable_mujoco_keys(self,window, key, scancode, action, mods):
    # 这里不执行任何操作，从而屏蔽默认快捷键
        pass
    
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
        geometry_solution = solveik.solve_ik_geometry(target_pos)
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
        
    def run(self):
        """主循环"""
        # 记录程序运行时间
        last_update = time.time()
        last_print_time = time.time()
        # 进入程序主循环
        while self.viewer.is_alive if self.viewer else True:

            # 输出控制循环耗时起点
            loop_start_time = time.time()
            
            # 处理控制器线程的交互,得到当前控制器的输入
            trans = controller.update_data()
            
            """前三轴逆解控制模式"""
            # 获取当前末端执行器的位置
            current_pos = self.data.xpos[self.end_effector_id].copy()
            
            # 仅在有输入的情况下进行逆解
            if np.any(np.abs(trans) > 1e-5):
                current_pos += trans
                self.solve_ik(current_pos)
                
            # 控制渲染更新频率
            if (time.time() - last_update) > 0.02:  # 50Hz
                self.viewer.render()
                last_update = time.time()

            # 输出控制循环耗时
            loop_interval = time.time()-loop_start_time
            
            # 打印解偏差
            if (time.time() - last_print_time) > 5:
                qpos_0 = f"{math.degrees(self.data.qpos[0]):.6f}"
                qpos_1 = f"{math.degrees(self.data.qpos[1]):.6f}"
                qpos_4 = f"{math.degrees(self.data.qpos[4]):.6f}"
                print(f"当前位置：{current_pos}")
                print(f"仿真角度：[{qpos_0}, {qpos_1}, {qpos_4}]")
                print(f"控制循环耗时: {(loop_interval) * 1000:.4f}ms")
                last_print_time = time.time()
        
        # 程序结束时关闭控制器线程
        controller.stop()
        debuggui.stop()
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
    ArmMJCcontroller = RobotArmController(model_path)
    # 调用控制器的 run 方法，开始主循环
    ArmMJCcontroller.run()