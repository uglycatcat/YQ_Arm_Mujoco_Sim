import pygame
import numpy as np
from threading import Thread, Event, Lock
import time

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

    def update_data(self, data):
        """更新数据，检查长度是否为10，否则报错并保持上一次数据"""
        if len(data) != 10:
            print(f"⚠️ 数据长度应为10，但收到 {len(data)}，保持上一次数据")
            return  # 不更新数据
        
        with self.lock:
            self.point_data = data
            self.last_valid_data = data  # 存储有效数据

    def draw_figure(self):
        """绘制图形"""
        if self.point_data is None:
            return  # 无数据时不绘制
        
        self.screen.fill((255, 255, 255))  # 白色背景
        
        # 绘制坐标轴
        pygame.draw.line(self.screen, (200, 200, 200), (0, self.height//2), (self.width, self.height//2), 1)  # X轴
        pygame.draw.line(self.screen, (200, 200, 200), (self.width//2, 0), (self.width//2, self.height), 1)  # Y轴
        
        # 绘制点并连接
        points = []
        for i in range(0, len(self.point_data), 2):
            x = int(self.point_data[i] * self.scale + self.width//2)
            y = int(-self.point_data[i+1] * self.scale + self.height//2)
            points.append((x, y))
            pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 5)  # 红点
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, (0, 0, 255), False, points, 2)  # 蓝线连接
        
        pygame.display.flip()  # 更新显示

    def run_loop(self):
        """Pygame 主循环（运行在子线程）"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("机械臂关节位置图")
        
        while self.running.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running.clear()
                    break
            
            # 如果当前数据无效，使用上一次的有效数据
            if self.point_data is None and self.last_valid_data is not None:
                with self.lock:
                    self.point_data = self.last_valid_data
            
            self.draw_figure()
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