import socket
 
def find_free_port(host='127.0.0.1', base_port=0):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for port in range(1024+base_port, 65535):
            try:
                s.bind((host, port))
                return str(port)
            except socket.error as e:
                continue
            
    raise Exception("No free ports available!")