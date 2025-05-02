import numpy as np
import math
import time
from SolveIK import solveik
from scipy.spatial.distance import euclidean  # 用于计算两点间距离

class ArmMotionTrajectory:
    
    def __init__(self):
        # 在protocol文件中，进行电机编码器采样的结果以n行3列的矩阵形式存储在这里
        self.sampling_encoder_buffer=np.empty((0, 3), dtype=np.float32)
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
        # 存储计算开始时的时间
        compute_start_time=time.time()
        
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
        
        print(f"直线插补计算耗时{(time.time()-compute_start_time)*1000:.4f}ms")
        return np.array(trajectory_ik_angle)
    
    def bezier_interpolation(self):
        """贝塞尔曲线插补，根据每4个控制点生成一段三阶贝塞尔曲线"""
        compute_start_time = time.time()
        points = self.sampling_mjc_pos_buffer
        interpolated_points = np.empty((0, 3), dtype=np.float32)

        # 至少需要4个控制点
        if len(points) < 4:
            print("控制点不足，贝塞尔插补至少需要4个点")
            return None

        # 每4个点生成一段曲线，步长决定插值点数
        for i in range(0, len(points) - 3, 3):  # 可用滑窗推进，避免断裂
            P0, P1, P2, P3 = points[i], points[i + 1], points[i + 2], points[i + 3]
            curve = []

            # 插补点数根据估算长度决定
            chord_length = euclidean(P0, P3)
            num_points = max(2, int(chord_length / self.interpolation_step) + 1)

            for t in np.linspace(0, 1, num_points):
                B_t = ((1 - t) ** 3) * P0 + \
                    3 * ((1 - t) ** 2) * t * P1 + \
                    3 * (1 - t) * (t ** 2) * P2 + \
                    (t ** 3) * P3
                curve.append(B_t)

            interpolated_points = np.vstack((interpolated_points, curve))

        # 对每个插补点进行逆解计算
        trajectory_ik_angle = []
        for point in interpolated_points:
            angle = solveik.solve_ik_geometry(point)
            trajectory_ik_angle.append(angle)

        print(f"贝塞尔插补计算耗时 {(time.time() - compute_start_time) * 1000:.4f}ms")
        return np.array(trajectory_ik_angle)
        
    def add_data(self):
        
        pass
        
    def delete_data(self):
        
        pass
        
    def change_data(self):
        
        pass
        
        return None
    
trajectory = ArmMotionTrajectory()