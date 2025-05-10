# client.py
# 在ubuntu上启动服务器后，可以通过此文件进行简单测试
import socket
import sys

def send_message():
    host = '10.5.5.1'  # Ubuntu的AP IP
    port = 5000        # 与服务端端口一致
    
    try:
        # 创建socket并设置超时（5秒连接超时，10秒发送超时）
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)  # 连接超时设置
            
            print(f"尝试连接到服务端 {host}:{port}...")
            
            try:
                s.connect((host, port))
                print("连接成功！输入消息发送（输入'quit'退出）")
                s.settimeout(10)  # 发送超时设置
                
                while True:
                    try:
                        message = input("> ")
                        if message.lower() == 'quit':
                            print("正在断开连接...")
                            break
                            
                        s.sendall(message.encode('utf-8'))
                        print(f"[已发送] {message}")
                        
                    except socket.timeout:
                        print("错误：发送超时，请检查网络连接")
                        break
                    except KeyboardInterrupt:
                        print("\n用户中断操作")
                        break
                        
            except socket.timeout:
                print(f"错误：连接超时，请检查：")
                print(f"1. 服务端IP是否正确（当前: {host}）")
                print(f"2. 服务端程序是否运行")
                print(f"3. 防火墙是否放行端口 {port}（sudo ufw allow {port}）")
            except ConnectionRefusedError:
                print(f"错误：连接被拒绝，请确认服务端已在 {host}:{port} 启动")
            except Exception as e:
                print(f"未知连接错误: {str(e)}")
                
    except socket.gaierror:
        print("错误：无效的主机地址，请检查IP配置")
    except Exception as e:
        print(f"程序异常: {str(e)}")
    finally:
        print("客户端已关闭")

if __name__ == "__main__":
    send_message()
    sys.exit(0)