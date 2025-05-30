# 测试性质文件，随时可以删除
import numpy as np
import mujoco as mj
import mujoco_viewer
import glfw
import time
import math
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from SolveIK import solveik
from Controller import controller
from Trajectory import trajectory

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
        
        self.motion_trajectory = trajectory.optimize_trajectory()
        
        controller.start()
        
    def disable_mujoco_keys(self,window, key, scancode, action, mods):
    # 这里不执行任何操作，从而屏蔽默认快捷键
        pass
    
    def refrash_joint(self,theta_1, theta_2, theta_3):
        # 更新前三个关节（0,1,4）
        self.data.qpos[0] = theta_1
        self.data.qpos[1] = theta_2
        self.data.qpos[4] = theta_3
        self.data.qpos[2] = self.data.qpos[1]
        self.data.qpos[3] = self.data.qpos[1]
        self.data.qpos[5] = self.data.qpos[4]  # 根据实际机械结构调整
        self.data.qpos[6] = -self.data.qpos[5]
        # 保持最后三个关节为0
        self.data.qpos[7] = 0
        self.data.qpos[8] = 0
        self.data.qpos[9] = 0
        
    def run(self):
        """主循环"""
        while self.viewer.is_alive if self.viewer else True:
            
            # 记录程序运行时间
            last_update = time.time()
            # 用于轨迹补点
            i=0
            
            print("插补点数:", self.motion_trajectory.shape[0])
            while(i<self.motion_trajectory.shape[0]):
                # 取出当前插补点的关节角
                theta_1, theta_2, theta_3 = self.motion_trajectory[i]
                self.refrash_joint(theta_1, theta_2, theta_3)
                # 执行前向动力学计算
                mj.mj_forward(self.model, self.data)
                mj.mj_step(self.model, self.data)
                # 短暂延时
                time.sleep(0.02)
                self.viewer.render()
                # 移动序号
                i+=1
                
            print("轨迹跟踪运行时长:", time.time() - last_update)
            
        # 程序结束时关闭控制器线程
        controller.stop()
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