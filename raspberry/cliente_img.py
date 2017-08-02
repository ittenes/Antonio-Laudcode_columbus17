#!/usr/bin/python

'''CLIENTE'''

import socket
import cv2
import numpy

TCP_IP = '192.168.1.36'
TCP_PORT = 5001

s = socket.socket()
s.connect((TCP_IP, TCP_PORT))


capture = cv2.VideoCapture(0)

while (True):

    ret, frame = capture.read()

    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()

    s.send( str(len(stringData)).ljust(16))
    s.send( stringData )

    a = s.recv(1024)
    print a
sock.close()
