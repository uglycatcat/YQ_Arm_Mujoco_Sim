import numpy as np
import math
import time
from SolveIK import solveik
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean  # 用于计算两点间距离
from scipy.interpolate import make_interp_spline # 用于计算样条曲线

class ArmMotionTrajectory:
    
    def __init__(self):
        # 在protocol文件中，进行电机编码器采样的结果以n行3列的矩阵形式存储在这里 目前不使用
        self.sampling_encoder_buffer=np.empty((0, 3), dtype=np.float32)
        # 测试用变量组
        self.sampling_test_buffer=np.empty((0, 3), dtype=np.float32)
        # 在main文件中，采样时进行采样的目标点存放在此处。目标控制点三维坐标以n行3列的矩阵形式存储在这里
        self.sampling_mjc_pos_buffer=np.empty((0, 3), dtype=np.float32)
        # 填充测试用数据，一共六个控制点，坐标单位为m
        self.sampling_test_buffer = np.vstack((
            self.sampling_test_buffer,
            np.array([-0.32774725, -0.51468068, 0.43656295], dtype=np.float32).reshape(1, -1),
            np.array([-0.33578701, -0.36444909, 0.4445864], dtype=np.float32).reshape(1, -1),
            np.array([-0.49604285, -0.09849478, 0.44389422], dtype=np.float32).reshape(1, -1),
            np.array([-0.32721762, 0.19992613, 0.30882229], dtype=np.float32).reshape(1, -1),
            np.array([-0.0541014, 0.30229002, 0.41897613], dtype=np.float32).reshape(1, -1),
            np.array([0.30646222, 0.15732615, 0.25583029], dtype=np.float32).reshape(1, -1)
        ))
        
    def linear_interpolation(self):
        """基于五次多项式的轨迹规划，控制起止速度/加速度为0，并按等时间间隔进行插补"""
        # 边界条件控制
        if len(self.sampling_mjc_pos_buffer)<2:
            print("直线插值至少需要两个点")
            return
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
    
    def smooth_global_interpolation(self):
        """五次样条轨迹插值,曲线通过所有控制点,整体路径二阶可导（连续加速度）"""
        # 参数定义
        v_max = 0.4      # 最大速度 (m/s)
        a_max = 0.2      # 最大加速度 (m/s^2)
        interval_time = 0.02  # 插补时间间隔 (s)
        positions = self.sampling_test_buffer  # 控制点，N x 3
        
        # 边界条件控制
        if len(positions)<3:
            print("样条曲线插值至少需要三个点")
            return
        
        # 存储计算开始时的时间
        compute_start_time = time.time()
        
        # 计算每一段距离与时间
        segment_times = []
        for i in range(len(positions) - 1):
            d = euclidean(positions[i], positions[i+1])
            T_v = 15 * d / (8 * v_max)
            T_a = np.sqrt(10 * d / a_max)
            T = max(T_v, T_a)
            segment_times.append(T)
            
        # 构造累计时间戳
        time_stamps = [0]
        for T in segment_times:
            time_stamps.append(time_stamps[-1] + T)
        time_stamps = np.array(time_stamps)
        
        # 构造样条（使用 make_interp_spline, k=5 表示五次）
        positions = np.array(positions)
        spline_x = make_interp_spline(time_stamps, positions[:,0], k=5, bc_type=([ (1, 0.0), (2, 0.0) ], [ (1, 0.0), (2, 0.0) ]))
        spline_y = make_interp_spline(time_stamps, positions[:,1], k=5, bc_type=([ (1, 0.0), (2, 0.0) ], [ (1, 0.0), (2, 0.0) ]))
        spline_z = make_interp_spline(time_stamps, positions[:,2], k=5, bc_type=([ (1, 0.0), (2, 0.0) ], [ (1, 0.0), (2, 0.0) ]))
        
        # 构造插值时间序列
        total_time = time_stamps[-1]
        t_array = np.arange(0, total_time, interval_time)
        if t_array[-1] < total_time:
            t_array = np.append(t_array, total_time)

        # 插值计算
        interpolated_points = np.stack([
            spline_x(t_array),
            spline_y(t_array),
            spline_z(t_array)
        ], axis=1)
        
        # 对每个插补点进行逆解计算
        trajectory_ik_angle = []
        for point in interpolated_points:
            angle = solveik.solve_ik_geometry(point)
            trajectory_ik_angle.append(angle)

        print(f"五次样条计算耗时 {(time.time() - compute_start_time) * 1000:.4f}ms")
        return np.array(trajectory_ik_angle)
    
    def smooth_global_interpolation_const_velocity(self):
        """五次样条轨迹插值（恒定速度插值）"""
        v_const = 0.1              # 恒定速度 (m/s)
        interval_time = 0.02       # 插补间隔 (s)
        positions = self.sampling_test_buffer  # N x 3 控制点

        if len(positions) < 3:
            print("样条曲线插值至少需要三个点")
            return

        compute_start_time = time.time()

        # 1. 计算累计路径长度
        positions = np.array(positions)
        distances = [0.0]
        for i in range(1, len(positions)):
            d = euclidean(positions[i], positions[i - 1])
            distances.append(distances[-1] + d)
        arc_lengths = np.array(distances)

        # 2. 构造基于路径长度 s 的样条曲线
        spline_x = make_interp_spline(arc_lengths, positions[:, 0], k=5)
        spline_y = make_interp_spline(arc_lengths, positions[:, 1], k=5)
        spline_z = make_interp_spline(arc_lengths, positions[:, 2], k=5)

        # 3. 生成匀速采样的路径长度数组
        total_length = arc_lengths[-1]
        num_points = int(np.ceil(total_length / (v_const * interval_time))) + 1
        s_samples = np.linspace(0, total_length, num_points)

        # 4. 生成插值点
        interpolated_points = np.stack([
            spline_x(s_samples),
            spline_y(s_samples),
            spline_z(s_samples)
        ], axis=1)

        # 5. 对每个插值点求逆解
        trajectory_ik_angle = []
        for point in interpolated_points:
            angle = solveik.solve_ik_geometry(point)
            trajectory_ik_angle.append(angle)

        print(f"恒速插值计算耗时 {(time.time() - compute_start_time) * 1000:.4f}ms")
        return np.array(trajectory_ik_angle)
    
    def optimize_trajectory(self):
        """优化轨迹生成（目标为最小控制力），输入为空间路径，输出为关节角轨迹"""
        positions = self.sampling_test_buffer  # N x 3 空间控制点

        if len(positions) < 2:
            print("优化轨迹至少需要两个点")
            return

        N = 100       # 每段插值点数
        dt = 0.02    # 插补时间间隔 (s)
        v_max = 0.4  # 最大关节速度 (rad/s)
        a_max = 0.2  # 最大关节加速度 (rad/s^2)

        compute_start_time = time.time()
        trajectory_ik_angle = []

        for i in range(len(positions) - 1):
            # === 1. 用逆解获取两点的关节角度 ===
            pose_start = positions[i]
            pose_goal = positions[i + 1]

            theta_start = solveik.solve_ik_geometry(pose_start)  # 1 x DOF
            theta_goal = solveik.solve_ik_geometry(pose_goal)    # 1 x DOF

            if theta_start is None or theta_goal is None:
                print(f"第{i}段逆解失败，跳过")
                continue

            dof = len(theta_start)

            # === 2. 初始猜测：线性插值 ===
            theta_init = np.linspace(theta_start, theta_goal, N)
            x0 = theta_init.flatten()  # 优化变量：N × DOF 扁平化

            # === 3. 优化目标函数 ===
            def objective(x):
                theta = x.reshape(N, dof)
                acc = theta[2:] - 2 * theta[1:-1] + theta[:-2]
                return np.sum(acc**2)

            # === 4. 构造约束 ===
            constraints = []

            # 起点终点约束
            def start_constraint(x): return x[:dof] - theta_start
            def goal_constraint(x): return x[-dof:] - theta_goal
            constraints.append({'type': 'eq', 'fun': start_constraint})
            constraints.append({'type': 'eq', 'fun': goal_constraint})

            # 速度约束
            def velocity_constraint(x):
                theta = x.reshape(N, dof)
                vel = (theta[1:] - theta[:-1]) / dt
                return v_max - np.abs(vel).flatten()

            # 加速度约束
            def accel_constraint(x):
                theta = x.reshape(N, dof)
                acc = (theta[2:] - 2 * theta[1:-1] + theta[:-2]) / dt**2
                return a_max - np.abs(acc).flatten()

            constraints.append({'type': 'ineq', 'fun': velocity_constraint})
            constraints.append({'type': 'ineq', 'fun': accel_constraint})

            # === 5. 优化执行 ===
            result = minimize(objective, x0, constraints=constraints, method='SLSQP')

            if not result.success:
                print(f"第{i}段优化失败: {result.message}")
                continue

            # === 6. 获取优化结果 ===
            theta_traj = result.x.reshape(N, dof)
            trajectory_ik_angle.extend(theta_traj.tolist())

        print(f"优化轨迹计算耗时 {(time.time() - compute_start_time) * 1000:.4f}ms")
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