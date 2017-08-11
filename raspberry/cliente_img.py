#!/usr/bin/python

'''CLIENTE'''

import socket
import cv2
import numpy

TCP_IP = '192.168.1.36'
TCP_PORT = 5001

s = socket.socket()
s.connect((TCP_IP, TCP_PORT))

w = 1280
h = 1024
capture = cv2.VideoCapture(0)
capture.set(3,w)
capture.set(4,h)

while (True):

    ret, frame = capture.read()

    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()

    decimg=cv2.imdecode(data,1)
    cv2.imshow('CLIENT',decimg)


    s.send( str(len(stringData)).ljust(16))
    s.send( stringData )

    a = s.recv(1024)
    print a









sock.close()
