#!/usr/bin/python

'''CLIENTE'''
#
# import socket
# import cv2
# import numpy
#
# TCP_IP = 'localhost'
# TCP_PORT = 5001
#
# sock = socket.socket()
# sock.connect((TCP_IP, TCP_PORT))
#
# capture = cv2.VideoCapture(0)
# ret, frame = capture.read()
#
# encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
# result, imgencode = cv2.imencode('.jpg', frame, encode_param)
# data = numpy.array(imgencode)
# stringData = data.tostring()
#
# sock.send( str(len(stringData)).ljust(16));
# sock.send( stringData );
# sock.close()
#
# decimg=cv2.imdecode(data,1)
# cv2.imshow('CLIENT',decimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


'''SERVER'''

#!/usr/bin/python
import socket
import cv2
import numpy

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

TCP_IP = '192.168.1.36'
TCP_PORT = 5001

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
conn, addr = s.accept()

length = recvall(conn,16)
stringData = recvall(conn, int(length))
data = numpy.fromstring(stringData, dtype='uint8')
s.close()

decimg=cv2.imdecode(data,1)
# cv2.imshow('SERVER',decimg)
cv2.imwrite('waka.jpg', decimg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
