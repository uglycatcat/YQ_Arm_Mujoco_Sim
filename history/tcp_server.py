# server.py
import socket

def start_server():
    host = '10.5.5.1'  # Ubuntu的AP IP
    port = 5000        # 选择一个空闲端口

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        print(f"Server listening on {host}:{port}...")

        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print("Received:", data.decode('utf-8'))

if __name__ == "__main__":
    start_server()