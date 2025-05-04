# 目的是优化其中的机械臂逆解算法
# 本文件使用几何解逆解将得到的结果传给机械臂
# 同时使用Xbox或者键盘接收信息
# 通过串口协议将信息传递给下位机
# 本文件是相对于main.py的几何解优化版本
# 计算复杂度由O（n*k）降低为O（1）
# sudo /home/sunrise/miniconda3/envs/mujoco_env/bin/python main/main.py
import numpy as np
import mujoco as mj
import mujoco_viewer
import glfw
import time
import math
from pathlib import Path
# 引入自定义串口协议
from Protocol import protocol
# 引入逆解相关运算
from SolveIK import solveik
# 引入控制器和控制模式
from Controller import controller
# 引入轨迹控制的相关函数
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
        # 打印帮助信息，显示控制器的使用说明
        controller.help()
        # 启动控制器线程
        controller.start()
        # 启动串口通信协议线程
        protocol.start()
        
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
    
    def sampling_command(self):

        val1 = self.data.qpos[0]
        val2 = self.data.qpos[1]
        val3 = self.data.qpos[4]
        raw_value=[val1,val2,val3]

        # 转成 numpy 行向量并拼接
        new_row = np.array(raw_value).reshape(1, -1)
        trajectory.sampling_encoder_buffer = np.vstack((trajectory.sampling_encoder_buffer, new_row))

        print(f"内部采样成功: {raw_value}")        
        
    def run(self):
        """主循环"""
        # 计算轨迹
        trajectory1=trajectory.linear_interpolation()
        Trajectory2=trajectory.smooth_global_interpolation()
        # 记录程序运行时间
        last_update = time.time()
        last_print_time = time.time()
        # 初始化当前控制模式
        current_control_mode = controller.update_mode()
        # 进入程序主循环
        while self.viewer.is_alive if self.viewer else True:
            
            # 用于轨迹补点
            i=0
            # 输出控制循环耗时起点
            loop_start_time = time.time()
            
            # 处理控制器线程的交互,得到当前控制器的输入
            trans = controller.update_data()
            
            # 处理控制器线程的交互,处理当前是否有特殊命令（比如打点采样）
            if controller.command==1: 
                protocol.sampling_command()
                controller.command=0;
            if controller.command==2: 
                self.sampling_command()
                controller.command=0;
                
            # 处理控制器线程的交互,处理当前是否发生了控制模式的变化
            if current_control_mode != controller.update_mode(): 
                current_control_mode=controller.update_mode()
                protocol.change_mode(current_control_mode)    
            
            # 根据控制模式判断当前任务
            if current_control_mode==13:
                """前三轴逆解控制模式"""
                # 获取当前末端执行器的位置
                current_pos = self.data.xpos[self.end_effector_id].copy()
                
                # 仅在有输入的情况下进行逆解
                if np.any(np.abs(trans) > 1e-5):
                    current_pos += trans
                    self.solve_ik(current_pos)

                # 更新关节角度到串口协议
                protocol.update_angles([self.data.qpos[i] for i in self.control_list])
            else:
                """轨迹控制模式"""
                print("插补点数:", Trajectory2.shape[0])
                
                while(i<Trajectory2.shape[0]):
                    # 取出当前插补点的关节角
                    theta_1, theta_2, theta_3 = Trajectory2[i]
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
                    # 执行前向动力学计算
                    mj.mj_forward(self.model, self.data)
                    mj.mj_step(self.model, self.data)
                    # 短暂延时
                    time.sleep(0.02)
                    self.viewer.render()
                    # 移动序号
                    i+=1
                
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
    ArmMJCcontroller = RobotArmController(model_path)
    # 调用控制器的 run 方法，开始主循环
    ArmMJCcontroller.run()