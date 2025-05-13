import pygame
import threading
import time

class ControllerDebugGUI:
    
    def __init__(self):
        """实例初始化"""
        self.width = 800
        self.height = 600
        self.control_motor_dict = {
            "Left_Hip_Roll": 0,
            "Left_Hip_Pitch": 0,
            "Left_Knee_Pitch": 0,
            "Left_Ankle_Wheel": 0,
            "Right_Hip_Roll": 0,
            "Right_Hip_Pitch": 0,
            "Right_Knee_Pitch": 0,
            "Right_Ankle_Wheel": 0,
        }
        self.running = threading.Event()
        self.thread = None
        self.screen = None
        self.font = None
        self.init_background()
        
    def init_background(self):
        """初始化背景"""
        self.background = pygame.Surface((self.width, self.height))
        self.background.fill((255, 255, 255))
        
    def show_control_data(self):
        """将控制数据control_motor_dict绘制在窗口中"""
        if self.screen is None or self.font is None:
            return
            
        # 清屏
        self.screen.blit(self.background, (0, 0))
        
        # 绘制标题
        title = self.font.render("机器人控制数据", True, (0, 0, 0))
        self.screen.blit(title, (self.width//2 - title.get_width()//2, 20))
        
        # 绘制每个关节的数据
        y_pos = 80
        for joint, value in self.control_motor_dict.items():
            text = self.font.render(f"{joint}: {value:.2f}", True, (0, 0, 0))
            self.screen.blit(text, (50, y_pos))
            y_pos += 40
            
        pygame.display.flip()
        
    def start(self):
        """启动GUI线程"""
        if not self.running.is_set():
            self.running.set()
            self.thread = threading.Thread(target=self.run_loop)
            self.thread.daemon = True  # 设为守护线程，主线程退出时自动结束
            self.thread.start()
        
    def stop(self):
        """关闭GUI线程"""
        self.running.clear()
        if self.thread is not None:
            self.thread.join(timeout=1)
        pygame.quit()
        
    def run_loop(self):
        """核心GUI程序(运行在子线程)"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("控制器数据可视化")
        self.font = pygame.font.SysFont('Microsoft YaHei', 24)
        
        while self.running.is_set():
            start_time = time.time()
            
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running.clear()
                    break
                    
            # 显示数据
            self.show_control_data()
            
            # 保持50Hz刷新率
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.02 - elapsed)
            time.sleep(sleep_time)
            
        pygame.quit()
        
    def receive_data(self, control_data):
        """获取机器人的控制数据数组"""
        if len(control_data) != 8: 
            print("传入GUI的关节数组长度不正确")
            return
            
        # 如果数据格式正常，将control_data按顺序传入字典
        keys = list(self.control_motor_dict.keys())
        for i in range(8):
            self.control_motor_dict[keys[i]] = control_data[i]
            
    def __del__(self):
        """析构函数"""
        self.stop()

# 创建全局实例
debuggui = ControllerDebugGUI()