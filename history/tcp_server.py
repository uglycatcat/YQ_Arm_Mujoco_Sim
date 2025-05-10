# 用于ESP32与RDK X3(Ubuntu)进行TCP通信的TCP服务端守护线程
# 测试版本
import signal
import sys
import socket
import threading
import re
import time
from threading import Event

class Esp32Communication():
    def __init__(self):
        """构造函数"""
        self.host = '10.5.5.1'  # Ubuntu的AP IP
        self.port = 5000        # TCP端口
        self.sock = None
        self.conn = None
        self.addr = None
        self._stop_event = Event()
        self._thread = threading.Thread(target=self.run_loop, daemon=True)
        self._lock = threading.Lock()  # 线程锁保证资源安全

    def start(self):
        """启动线程"""
        if not self._thread.is_alive():
            self._stop_event.clear()
            try:
                # 初始化TCP套接字
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sock.bind((self.host, self.port))
                self.sock.listen(1)
                print(f"[TCP服务端] 监听 {self.host}:{self.port}")
            except Exception as e:
                print(f"初始化失败: {str(e)}")
                self.__del__()
                return
            self._thread.start()
            print("[TCP服务端] 守护线程已启动")

    def stop(self):
        """停止线程"""
        if self._thread.is_alive():
            self._stop_event.set()
            # 关闭套接字强制退出accept阻塞
            with self._lock:
                if self.conn:
                    self.conn.close()
                if self.sock:
                    self.sock.close()
            print("[TCP服务端] 正在停止...")

    def run_loop(self):
        """线程主循环"""
        while not self._stop_event.is_set():
            try:
                # 接受新连接（带超时检测）
                self.sock.settimeout(10)  # 每秒检测一次停止标志
                print("[TCP服务端] 等待设备连接...")
                self.conn, self.addr = self.sock.accept()
            except socket.timeout:
                continue  # 正常超时，继续循环检测停止标志
            except OSError as e:
                if self._stop_event.is_set():
                    break  # 主动停止导致的错误无需处理
                print(f"接受连接异常: {str(e)}")
                continue

            with self.conn:
                print(f"[TCP服务端] 设备 {self.addr} 已连接")
                self.conn.settimeout(2)  # 数据接收超时时间
                while not self._stop_event.is_set():
                    try:
                        data = self.conn.recv(1024)
                        if not data:
                            print(f"[TCP服务端] 设备 {self.addr} 断开连接")
                            break
                        self.handle_input(data)
                    except socket.timeout:
                        continue  # 超时继续检测停止标志
                    except Exception as e:
                        print(f"[TCP服务端] 数据接收异常: {str(e)}")
                        break
            print(f"[TCP服务端] 连接 {self.addr} 已关闭")

    def handle_input(self, data):
        """处理传入数据（需根据业务需求自定义）"""
        try:
            # 示例：打印UTF-8解码后的内容
            decoded_data = data.decode('utf-8').strip()
            print(f"[TCP数据] 收到消息: {decoded_data}")
            
            # 处理数据
            last_data=decoded_data[-1]
            if "add" in decoded_data:
                self.sampling_point(last_data)
            if "del" in decoded_data:
                self.delete_point(last_data)
            if "but3" in decoded_data:
                self.motion_trigger(last_data)
            if "but4" in decoded_data:
                self.video_record(last_data)
            if "JS1" in decoded_data:
                if "CENTER" in decoded_data:
                    self.set_joystick_js1(0, 0)
                elif match := re.search(r"X=(\d{1,2}) Y=(\d{1,2})", decoded_data):
                    self.set_joystick_js1(*map(int, match.groups()))
            if "JS2" in decoded_data:
                if "CENTER" in decoded_data:
                    self.set_joystick_js2(0, 0)
                elif match := re.search(r"X=(\d{1,2}) Y=(\d{1,2})", decoded_data):
                    self.set_joystick_js2(*map(int, match.groups()))
                
        except UnicodeDecodeError:
            print(f"[TCP数据] 收到原始字节: {data.hex()}")

    def send_response(self, message):
        """发送响应（线程安全）"""
        with self._lock:
            if self.conn and not self._stop_event.is_set():
                try:
                    self.conn.sendall(message.encode('utf-8'))
                    print(f"[TCP发送] 已发送: {message}")
                except Exception as e:
                    print(f"[TCP发送] 发送失败: {str(e)}")

    def __del__(self):
        """析构函数"""
        self.stop()
        time.sleep(0.5)  # 等待资源释放
        print("[TCP服务端] 资源已清理")
    
    def sampling_point(self,data):
        print(data)
        return
    
    def delete_point(self,data):
        print(data)
        return
    
    def motion_trigger(self,data):
        print(data)
        return
    
    def video_record(self,data):
        print(data)
        return
    
    def set_joystick_js1(self,x,y):
        print(x,y)
        return
    
    def set_joystick_js2(self,x,y):
        print(x,y)
        return

# 全局实例化
esp32tcpcom = Esp32Communication()

# 测试用
if __name__ == "__main__":
    
    # 优雅退出处理
    def graceful_exit(signum, frame):
        print("\n收到退出信号，正在清理...")
        esp32tcpcom.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, graceful_exit)  # 处理Ctrl+C
    signal.signal(signal.SIGTERM, graceful_exit) # 处理kill命令

    # 启动服务并保持主线程
    esp32tcpcom.start()
    print("服务运行中，按 Ctrl+C 停止")
    
    # 最简化的保持主线程方式
    signal.pause()  # 挂起主线程直到收到信号