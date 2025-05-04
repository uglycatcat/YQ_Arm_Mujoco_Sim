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
        
    def linear_interpolation(self):
        """基于五次多项式的轨迹规划，控制起止速度/加速度为0，并按等时间间隔进行插补"""
        # 参数定义
        v_max = 0.12      # 最大速度 (m/s)
        a_max = 0.15      # 最大加速度 (m/s^2)
        interval_time = 0.02  # 插补时间间隔 (s)
        
        # 插补点总数与参数关系markdown表达式
        # \[
        # N = \left\lceil \frac{1}{\text{interval\_time}} \cdot \max\left( \frac{15D}{8v_{\text{max}}}, \sqrt{\frac{10D}{a_{\text{max}}}} \right) \right\rceil + 1
        # \]
        
        # 存储计算开始时的时间
        compute_start_time = time.time()
        interpolated_points = []
        
        # 遍历每两个相邻的控制点
        for i in range(len(self.sampling_mjc_pos_buffer) - 1):
            p0 = self.sampling_mjc_pos_buffer[i]
            p1 = self.sampling_mjc_pos_buffer[i + 1]
            distance = euclidean(p0, p1)
            
            # 初步估计时间（保守估计）
            # T 满足最大速度和加速度的限制
            T_v = 15 * distance / (8 * v_max)
            T_a = np.sqrt(10 * distance / a_max)
            T = max(T_v, T_a)
            
            # 构造时间序列
            t_array = np.arange(0, T, interval_time)
            if t_array[-1] < T:
                t_array = np.append(t_array, T)
            
            # 分别对x, y, z计算系数
            coeffs = [self.compute_coeffs(p0[j], p1[j], T) for j in range(3)]
            
            
            # 生成插补点
            for t in t_array:
                point = np.array([
                    sum(c * t**n for n, c in enumerate(coeffs[dim]))
                    for dim in range(3)
                ])
                interpolated_points.append(point)
        
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
       
        # 插补步长 (单位: m)
        interpolation_step = 0.01  
        
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
            num_points = max(2, int(chord_length / interpolation_step) + 1)

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
    
    def compute_coeffs(self,p_start, p_end, T):
        # 起止点速度、加速度均为0
        a0 = p_start
        a1 = 0
        a2 = 0
        a3 = (10 * (p_end - p_start)) / T**3
        a4 = (-15 * (p_end - p_start)) / T**4
        a5 = (6 * (p_end - p_start)) / T**5
        return a0, a1, a2, a3, a4, a5
    
    def add_data(self):
        
        pass
        
    def delete_data(self):
        
        pass
        
    def change_data(self):
        
        pass
        
        return None
    
trajectory = ArmMotionTrajectory()