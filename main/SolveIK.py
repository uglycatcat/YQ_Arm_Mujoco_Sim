import numpy as np
import keyboard
import time
import math
from scipy.spatial.transform import Rotation as R

class SolveIKMethod:
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
    
    # def numerical_solution(self,target_pos，target_rot):
    #     """逆运动学求解，数值方法，此函数未做适配，可以参考history文件夹中的main_v1"""
    #     def objective(q):
    #         # 仅更新可控关节
    #         for i, joint_idx in enumerate(self.control_list):
    #             self.data.qpos[joint_idx] = q[i]
            
    #         self.data.qpos[2] = self.data.qpos[3] = self.data.qpos[1]  # joint3 = joint2_2
    #         self.data.qpos[5] = self.data.qpos[4]  # 根据实际机械结构调整
    #         self.data.qpos[6] = -self.data.qpos[5]
    #         mj.mj_forward(self.model, self.data)
            
    #         # 计算误差
    #         pos_err = np.linalg.norm(self.data.xpos[self.end_effector_id] - target_pos)
    #         orient_err = np.linalg.norm(self.data.xmat[self.end_effector_id].reshape(3, 3) - target_rot.as_matrix(), ord='fro') / 3
    #         return pos_err + 0.5 * orient_err  # 调整权重平衡
        
    #     # 关节角度约束
    #     constraints = [
    #         {"type": "ineq", "fun": lambda q, i=i: q[i] - self.model.jnt_range[self.control_list[i], 0]} for i in range(len(self.control_list))
    #     ] + [
    #         {"type": "ineq", "fun": lambda q, i=i: self.model.jnt_range[self.control_list[i], 1] - q[i]} for i in range(len(self.control_list))
    #     ]
        
    #     # 初始值
    #     q_init = [self.data.qpos[i] for i in self.control_list]
    #     res = minimize(objective, q_init, method='SLSQP', constraints=constraints)
    #     return res
    
solveik = SolveIKMethod()