import numpy as np
import time
import socket
import sys
import _thread as thread

# Workaround fro ROS Kinetic issue importing cv
import sys
try:
  sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
  pass
import cv2

class Server():
  """TCP IP communication server
  If automatic_port == True, will iterate over port until find a free one
  """
  def __init__(self, ip, port, automatic_port=True, wait_for_wp3=False):
    self.__size_message_length = 16  # Buffer size for the length
    max_connections_attempts = 5

    self.conn = None
    self.conn_wp3 = None

    # Start and connect to client
    self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if automatic_port:
      connected = False
      while (not connected) and (max_connections_attempts > 0):
        try:
          self.s.bind((ip, port))
          connected = True
        except:
          print("[Server]: Port", port, "already in use. Binding to port:", port+1)
          port += 1
          max_connections_attempts -= 1
      if not connected:
        print("[Server]: Error binding to adress!")
    else:
      self.s.bind((ip, port))

    self.s.listen(True)
    print("[Server]: Waiting for connections...")

    thread.start_new_thread(self.listenForClients, ())

    # clients_count = 0
    # while 1:
    #     conn, addr = self.s.accept()
    #     msg = self.receiveFrom(conn)
    #     print ("ID from client: ", msg)
    #     if msg == "WP2":
    #         self.conn = conn
    #         clients_count = clients_count + 1
    #         print("[Server]: Connected with WP 2: ", addr)

    #         if wait_for_wp3 == False:
    #             break
    #     elif msg == "WP3":
    #         self.conn_wp3 = conn
    #         clients_count = clients_count + 1
    #         print("[Server]: Connected with WP 3: ", addr)
    #     else:
    #         print("[Server]: Connected but unknown client: ", addr)
        
    #     if clients_count == 2:
    #         break
    
    # print("[Server]: Connected with all required clients !!")


  def listenForClients(self):
    while 1:
      try:
        conn, addr = self.s.accept()
        msg = self.receiveFrom(conn)
        print ("ID from client: ", msg)
        if msg == "WP2":
            self.conn = conn
            print("[Server]: Connected with WP 2: ", addr)
        elif msg == "WP3":
            self.conn_wp3 = conn
            print("[Server]: Connected with WP 3: ", addr)
        else:
            print("[Server]: Connected but unknown client: ", addr)
      except:
        print("Exception while listening from clients...")


  def __del__(self):
    self.s.close()

  def close(self):
    self.s.close()

  def isWP2Connected(self):
    return self.conn is not None

  def isWP3Connected(self):
    return self.conn_wp3 is not None

  def send(self, message):
    message_size = str(len(message)).ljust(self.__size_message_length).encode()
    self.conn.sendall(message_size)  # Send length of msg (in known size, 16)
    self.conn.sendall(message.encode())  # Send message

  def sendToWp3(self, message, encode=True):
    message_size = str(len(message)).ljust(self.__size_message_length).encode()
    self.conn_wp3.sendall(message_size)  # Send length of msg (in known size, 16)
    if encode == True:
      self.conn_wp3.sendall(message.encode())  # Send message
    else:
      self.conn_wp3.sendall(message)  # Send message without encode

  def receiveFrom(self, client_conn, decode=True):
    length = self.__receive_value(client_conn, self.__size_message_length)
    if length is not None:  # If None received, no new message to read
      message = self.__receive_value(client_conn, int(length), decode)  # Get message
      return message
    return None

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