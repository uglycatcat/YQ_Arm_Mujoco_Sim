# 本文件用于main_v2绘制机械臂平面图
# 将逆解结果可视化

import pygame
import numpy as np
from threading import Thread, Event, Lock
import time
import math
from numpy import degrees

class DrawPointFigure:
    def __init__(self):
        self.point_data = None
        self.last_valid_data = None  # 存储上一次有效数据
        self.lock = Lock()
        self.running = Event()
        self.screen = None
        self.clock = pygame.time.Clock()
        self.scale = 400  # 放大比例（原100，现改为400）
        self.width, self.height = 800, 600  # 窗口大小
        self.font = None  # 初始化为None
        self.font_size = 30  # 添加字体大小设置
        # 添加背景和坐标轴表面缓存
        self.background = None
        self.axes = None
        self._init_background()

    def _init_background(self):
        """初始化背景和坐标轴缓存"""
        self.background = pygame.Surface((self.width, self.height))
        self.background.fill((255, 255, 255))
        
        self.axes = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        # 只绘制Y轴大于等于0的部分
        pygame.draw.line(self.axes, (200, 200, 200), (0, self.height//2), (self.width, self.height//2), 1)  # X轴
        pygame.draw.line(self.axes, (200, 200, 200), (self.width//2, self.height//2), (self.width//2, 0), 1)  # Y轴（只画上半部分）

    def update_data(self, data):
        """更新数据，检查长度是否为10，否则报错并保持上一次数据"""
        if not isinstance(data, np.ndarray):
            print("⚠️ 数据必须是numpy数组")
            return
            
        if len(data) != 10:
            print(f"⚠️ 数据长度应为10，但收到 {len(data)}，保持上一次数据")
            return
            
        with self.lock:
            self.point_data = data.copy()  # 使用copy避免数据被修改
            self.last_valid_data = data.copy()

    def draw_figure(self):
        """绘制图形"""
        if self.point_data is None:
            return
            
        # 使用缓存的背景和坐标轴
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.axes, (0, 0))
        
        # 预计算中心点
        center_x, center_y = self.width//2, self.height//2
        
        # 绘制点并连接
        points = []
        for i in range(0, len(self.point_data), 2):
            x = int(self.point_data[i] * self.scale + center_x)
            y = int(-self.point_data[i+1] * self.scale + center_y)
            points.append((x, y))
            pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 5)
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, (0, 0, 255), False, points, 2)

    def show_data(self):
        """显示相邻点的ΔX、ΔY和距离"""
        if self.point_data is None:
            return

        # 调整起始位置和行间距
        text_y_offset = self.height // 2 + 10
        line_spacing = 25  # 减小行间距
        
        # 显示第一个点的坐标
        x1, y1 = self.point_data[0], self.point_data[1]
        text = f"Point 1: X={x1:.8f}, Y={y1:.8f}"
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, text_y_offset))
        text_y_offset += line_spacing
        
        # 显示中间点的差值
        for i in range(0, len(self.point_data)//2 - 1):
            x1, y1 = self.point_data[2*i], self.point_data[2*i+1]
            x2, y2 = self.point_data[2*(i+1)], self.point_data[2*(i+1)+1]
            
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx**2 + dy**2)
            
            text = f"Link{i+1} -> Link{i+2}: ΔX={dx:.8f}, ΔY={dy:.8f}, Distance={distance:.8f}"
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, text_y_offset))
            text_y_offset += line_spacing
            
            # 显示下一个点的坐标
            text = f"Point {i+2}: X={x2:.8f}, Y={y2:.8f}"
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, text_y_offset))
            text_y_offset += line_spacing
            
        # 显示最后一个点的逆解角度
        last_point = self.point_data[-2:]  # 获取最后两个值作为坐标
        theta_2, theta_3 = solve_ik_geometry_draw(last_point)
        text = f"IK Angles: θ2={np.degrees(theta_2):.2f}°, θ3={np.degrees(theta_3):.2f}°"
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, text_y_offset))

    def run_loop(self):
        """Pygame 主循环（运行在子线程）"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("机械臂关节位置图")
        self.font = pygame.font.SysFont(None, self.font_size)  # 使用设置的字体大小

        while self.running.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running.clear()
                    break

            # 如果当前数据无效，使用上一次的有效数据
            if self.point_data is None and self.last_valid_data is not None:
                with self.lock:
                    self.point_data = self.last_valid_data

            # 绘制图像和显示数据
            self.draw_figure()
            self.show_data()
            
            pygame.display.flip()  # 更新显示

            self.clock.tick(50)  # 50Hz刷新率

        pygame.quit()

    def start(self):
        """启动绘图线程"""
        self.running.set()
        self.thread = Thread(target=self.run_loop)
        self.thread.daemon = True  # 设为守护线程，主线程退出时自动结束
        self.thread.start()

    def stop(self):
        """停止绘图线程"""
        self.running.clear()
        if hasattr(self, "thread"):
            self.thread.join()  # 等待线程结束

# 全局实例
draw_figure = DrawPointFigure()

def solve_ik_geometry_draw(target_pos):
    """逆解几何计算(单位弧度制)"""
    
    # 计算P4相对于P1点的偏移量
    x_offset = (target_pos[0] - 0.078304) - 0.03419422 - 0.0162178
    y_offset = (target_pos[1] - 0.0044872) - 0.05481078 - 0.136133
    
    # 计算P4P1的距离
    distance = math.sqrt(x_offset**2 + y_offset**2)
    
    # P1P2和P3P4两段长度固定
    arm_length_1=0.4000
    arm_length_2=0.4000
    
    # 三边构成一个三角形
    # 计算theta_2（arm_length_1和distance构成的角）
    theta_2 = math.acos((arm_length_1**2 + distance**2 - arm_length_2**2) / (2 * arm_length_1 * distance))
    # 计算theta_3（arm_length_1和arm_length_2构成的角）
    theta_3 = math.acos((arm_length_1**2 + arm_length_2**2 - distance**2) / (2 * arm_length_1 * arm_length_2))
    
    # temp1得到P4P1两点的y/x的正切值对应的弧度制
    temp1 = math.atan2(y_offset, x_offset)
    
    # 根据计算结果得到真正的theta_2和theta_3
    theta_2 = temp1+theta_2-math.radians(120)
    theta_3 = (math.pi/2-(math.pi-theta_2-math.radians(120))+theta_3)-math.radians(70)
    
    # 返回弧度制的计算结果
    return theta_2, theta_3