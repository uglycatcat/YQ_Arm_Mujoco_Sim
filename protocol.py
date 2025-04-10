# 此处详细描述一下数据映射关系
# 关节角度以float类型存储，每个角度值对应四个字节
# 此处将每四个字节按照0-2pi映射到0-8191
# 其中前三个数据按顺序即为无刷电机1，2，3轴顺序
# 初始化六个数据都是0
# 一字节表示一轴：俯视视角逆时针转动数据增加
# 二字节表示二轴：向前倾数据增加
# 三字节表示三轴：向上仰数据增加
# 四字节表示YAW轴：俯视视角逆时针转动数据增加
# 五字节表示ROLL轴：注意云台横滚角无驱动，只做被动轴
# 六字节表示PITCH轴：向上仰数据增加
import serial
import struct
import time
from threading import Thread, Event, Lock
import numpy as np

class JointAngleProtocol:
    # CRC16 查表法的表
    CRC16_TABLE = [
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
        0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
        0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6,
        0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
        0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485,
        0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
        0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
        0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
        0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861, 0x2802, 0x3823,
        0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
        0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0x0a50, 0x3a33, 0x2a12,
        0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
        0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41,
        0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
        0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70,
        0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
        0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f,
        0x1080, 0x00a1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
        0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e,
        0x02b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
        0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
        0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
        0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c,
        0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
        0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab,
        0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3,
        0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a,
        0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
        0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9,
        0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0x0cc1,
        0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8,
        0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0x0ed1, 0x1ef0
    ]
    
#/dev/ttyUSB0
    def __init__(self, port="COM5", baudrate=115200):
        """初始化串口通信协议"""
        self.serial = None
        self.running = Event()
        self.joint_angles = np.zeros(6, dtype=np.float32)
        self.thread = None
        self.lock = Lock()
        self.seq = 0
        # 预分配缓冲区
        self.angles_buffer = bytearray(12)
        self.header_buffer = bytearray(8)
        self.packet_buffer = bytearray(22)
        
        # 尝试打开串口
        try:
            self.serial = serial.Serial(port, baudrate)
            self.serial_enabled = True
            print(f"成功连接到串口 {port}")
        except serial.SerialException as e:
            self.serial_enabled = False
            print(f"警告：无法连接到串口 {port}，将禁用串口功能")

    def calculate_crc16(self, data):
        """优化的CRC16计算"""
        crc = 0xFFFF
        table = self.CRC16_TABLE  # 本地变量访问更快
        for byte in data:
            crc = ((crc << 8) & 0xFF00) ^ table[((crc >> 8) ^ byte) & 0xFF]
        return crc & 0xFF, (crc >> 8) & 0xFF

    def start(self):
        """启动串口通信线程"""
        if not self.running.is_set():  # 避免重复启动
            self.running.set()
            self.thread = Thread(target=self._send_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """停止串口通信"""
        self.running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.serial and self.serial.is_open:
            self.serial.close()

    def update_angles(self, angles):
        """更新关节角度值"""
        with self.lock:
            np.copyto(self.joint_angles, angles)

    @staticmethod
    def _convert_angle_to_uint16(angle):
        """优化的角度转换"""
        return min(8191, int((angle % (2 * np.pi)) * 1303.8344))  # 8191/(2*pi) ≈ 1303.8344

    def _send_loop(self):
        """优化的发送循环"""
        interval = 0.02  # 50Hz
        next_send = time.monotonic()
        
        while self.running.is_set():
            try:
                current_time = time.monotonic()
                if current_time >= next_send:
                    self._send_packet()
                    next_send = current_time + interval
                    self.seq = (self.seq + 1) & 0xFFFF
                else:
                    # 使用更精确的休眠时间
                    sleep_time = next_send - current_time
                    if sleep_time > 0.001:  # 只对较长的休眠时间使用sleep
                        time.sleep(sleep_time * 0.8)
            except Exception as e:
                print(f"发送循环错误: {e}")
                time.sleep(0.1)

    def _send_packet(self):
        """优化的数据包发送"""
        if not self.serial_enabled:
            return
        
        try:
            # 获取当前角度值
            with self.lock:
                current_angles = self.joint_angles

            # 预分配缓冲区，避免重复创建
            angles_buffer = bytearray(12)
            header_buffer = bytearray(8)
            packet_buffer = bytearray(22)

            # 打包角度数据
            struct.pack_into('<HHHHHH', angles_buffer, 0,
                *(self._convert_angle_to_uint16(angle) for angle in current_angles))

            # 打包头部
            struct.pack_into('<HBHHB', header_buffer, 0,
                0xAAAA,     # 起始标志
                0x02,       # 控制字节
                12,        # 数据长度
                self.seq,   # 序号
                0x13       # 命令ID
            )

            # 组合数据并计算CRC
            packet_buffer[0:8] = header_buffer
            packet_buffer[8:20] = angles_buffer
            crc_low, crc_high = self.calculate_crc16(packet_buffer[0:20])
            packet_buffer[20] = crc_low
            packet_buffer[21] = crc_high

            # 发送数据
            self.serial.write(packet_buffer)
        except Exception as e:
            print(f"发送数据出错: {e}")
            self.serial_enabled = False
            print("串口通信已禁用")

# 全局实例
protocol = JointAngleProtocol()