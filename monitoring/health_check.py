import logging
import socket

def service_health_check(host="localhost", port=8000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

if __name__ == "__main__":
    if service_health_check():
        print("Service is healthy!")
    else:
        print("Service is down!")
