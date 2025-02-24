import socket
import threading
import select
import time
import sys

def handle_client(client_socket):
    """处理客户端连接"""
    request = client_socket.recv(4096)
    
    try:
        # 解析HTTP请求中的主机名和端口
        first_line = request.split(b'\n')[0]
        url = first_line.split(b' ')[1]
        
        http_pos = url.find(b'://')
        if http_pos == -1:
            temp = url
        else:
            temp = url[(http_pos + 3):]
            
        port_pos = temp.find(b':')
        host_pos = temp.find(b'/')
        
        if host_pos == -1:
            host_pos = len(temp)
            
        if port_pos == -1 or host_pos < port_pos:
            port = 80 if b'http://' in url else 443
            host = temp[:host_pos]
        else:
            port = int(temp[port_pos + 1:host_pos])
            host = temp[:port_pos]
            
        # 创建到目标服务器的连接
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.connect((host.decode(), port))
        
        if port == 443:
            # 对于HTTPS请求，发送连接建立成功的响应
            client_socket.send(b'HTTP/1.1 200 Connection established\r\n\r\n')
        else:
            # 对于HTTP请求，直接转发原始请求
            server_socket.send(request)
            
        # 在客户端和服务器之间转发数据
        while True:
            r, w, e = select.select([client_socket, server_socket], [], [], 1)
            if client_socket in r:
                data = client_socket.recv(4096)
                if not data:
                    break
                server_socket.send(data)
            if server_socket in r:
                data = server_socket.recv(4096)
                if not data:
                    break
                client_socket.send(data)
                
    except Exception as e:
        print(f"Error handling client: {e}")
        
    finally:
        client_socket.close()
        server_socket.close()

def main():
    try:
        # 创建代理服务器
        proxy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        proxy_socket.bind(('127.0.0.1', 8809))
        proxy_socket.listen(10)
        
        print("代理服务器运行在 127.0.0.1:8809")
        
        while True:
            client_socket, addr = proxy_socket.accept()
            print(f"接受连接来自: {addr}")
            
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket,)
            )
            client_thread.setDaemon(True)
            client_thread.start()
            
    except KeyboardInterrupt:
        print("\n关闭代理服务器...")
        proxy_socket.close()
        sys.exit(0)
        
if __name__ == '__main__':
    main() 