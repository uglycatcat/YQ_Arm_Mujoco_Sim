import numpy as np
import math
from SolveIK import solveik
from scipy.spatial.distance import euclidean  # 用于计算两点间距离

class ArmMotionTrajectory:
    
    def __init__(self):
        # 测试用变量，目标控制点三维坐标以n行3列的矩阵形式存储在这里
        self.sampling_mjc_pos_buffer=np.empty((0, 3), dtype=np.float32)
        # 填充测试用数据，一共六个控制点，坐标单位为m
        self.sampling_mjc_pos_buffer = np.vstack((
            self.sampling_mjc_pos_buffer,
            np.array([-0.32774725, -0.51468068, 0.43656295], dtype=np.float32).reshape(1, -1),
            np.array([-0.33578701, -0.36444909, 0.4445864], dtype=np.float32).reshape(1, -1),
            np.array([-0.49604285, -0.09849478, 0.44389422], dtype=np.float32).reshape(1, -1),
            np.array([-0.32721762, 0.19992613, 0.30882229], dtype=np.float32).reshape(1, -1),
            np.array([-0.0541014, 0.30229002, 0.41897613], dtype=np.float32).reshape(1, -1),
            np.array([0.30646222, 0.15732615, 0.25583029], dtype=np.float32).reshape(1, -1)
        ))
        # 设置插补参数
        self.interpolation_step = 0.01  # 插补步长 (单位: m)
        
    def linear_interpolation(self):
        """线性插补，对每两个控制点之间进行直线插补"""
        # 存储最终插补后的所有点
        interpolated_points = np.empty((0, 3), dtype=np.float32)
        
        # 遍历每两个相邻的控制点
        for i in range(len(self.sampling_mjc_pos_buffer) - 1):
            start_point = self.sampling_mjc_pos_buffer[i]
            end_point = self.sampling_mjc_pos_buffer[i + 1]
            
            # 计算两点间距离
            distance = euclidean(start_point, end_point)
            
            # 计算需要插补的点数 (至少插补1个点)
            num_points = max(2, int(distance / self.interpolation_step) + 1)
            
            # 在两个点之间进行线性插补
            for t in np.linspace(0, 1, num_points):
                interpolated_point = start_point + t * (end_point - start_point)
                interpolated_points = np.vstack((interpolated_points, interpolated_point))
        
        # 对每个插补点进行逆解计算
        trajectory_ik_angle = []
        for point in interpolated_points:
            # 逆解运算
            angle = solveik.solve_ik_geometry(point)
            trajectory_ik_angle.append(angle)
            
        return np.array(trajectory_ik_angle)
    
    def bessel_interpolation(self):
        """贝塞尔曲线插补，根据当前控制点的数量进行插补"""
        return None
        
    def add_data(self):
        
        pass
        
    def delete_data(self):
        
        pass
        
    def change_data(self):
        
        pass
        
        return None
    
trajectory = ArmMotionTrajectory()


        # # 在protocol文件中，进行电机编码器采样的结果以n行3列的矩阵形式存储在这里
        # self.sampling_encoder_buffer=np.empty((0, 3), dtype=np.float32)