import socket
from math import *
import pygame

sock = socket.socket()
sock.connect(('localhost', 9090))
sock.send(bytes('33', 'utf-8'))
sock.send(bytes('a', 'utf-8'))

data = sock.recv(1024)
sock.close()

print(data.decode('utf-8'))
