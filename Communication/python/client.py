import numpy as np
import time
import socket

import cv2

class Client():
  """TCP IP communication server
  """
  def __init__(self, host, port):
    self.__size_message_length = 16  # Buffer size for the length

    # Start and connect to server
    self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    max_connections_attempts = 5
    connected = False
    while (not connected) and (max_connections_attempts > 0):
        try:
            self.s.connect((host, port))
            connected = True
        except:
            max_connections_attempts -= 1
            print("[Client]: Failed to connect to: ", host, port, " Retrying...")
    
    print("[Client]: Connected with server...")
    self.conn = self.s

  def __del__(self):
    self.s.close()


    

  def send(self, message, encode=True):
    message_size = str(len(message)).ljust(self.__size_message_length).encode()
    self.conn.sendall(message_size)  # Send length of msg (in known size, 16)
    if encode == True:
      self.conn.sendall(message.encode())  # Send message
    else:
      self.conn.sendall(message)  # Send message without encode

  def receive(self, decode=True):
    length = self.__receive_value(self.conn, self.__size_message_length)
    if length is not None:  # If None received, no new message to read
      message = self.__receive_value(self.conn, int(length), decode)  # Get message
      return message
    return None




  def receive_image(self):
    data = self.receive(False)
    if data is not None:
      data = np.fromstring(data, dtype='uint8')
      decimg = cv2.imdecode(data, 1)
      return decimg
    return None

  def __receive_value(self, conn, buf_lentgh, decode=True):
    buf = b''
    while buf_lentgh:
      newbuf = conn.recv(buf_lentgh)
      # if not newbuf: return None  # Comment this to make it non-blocking
      buf += newbuf
      buf_lentgh -= len(newbuf)
    if decode:
      return buf.decode()
    else:
      return buf

  def clear_buffer(self):
    try:
      while self.conn.recv(1024): pass
    except:
      pass
    return