#!/usr/bin/python
import socket
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor, Adafruit_StepperMotor
import time
import atexit

import cv2
import numpy

TCP_IP = '192.168.1.36'
TCP_PORT = 5001
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


mh = Adafruit_MotorHAT()

# recommended for auto-disabling motors on shutdown!
def turnOffMotors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)

atexit.register(turnOffMotors)

myStepper1 = mh.getStepper(200, 1)  # 200 steps/rev, motor port #1
myStepper1.setSpeed(1600)             # 30 RPM
myStepper2 = mh.getStepper(200, 2)  # 200 steps/rev, motor port #1
myStepper2.setSpeed(1600)             # 30 RPM

length = None
buffer = ""
capture = cv2.VideoCapture(0)

while (True):

    #FOTO

    ret, frame = capture.read()

    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()

    sock.send( str(len(stringData)).ljust(16))
    sock.send( stringData )

    #DATOS

    data_pos = s.recv(50)
    buffer += data_pos

    length_str, ignored, buffer = buffer.rpartition(':')
    buffer, ignored, length = length_str.rpartition(':')
    print("Microsteps", length)
    myStepper1.step(10, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.MICROSTEP)
    myStepper1.step(10, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.MICROSTEP)
    myStepper2.step(10, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.MICROSTEP)
    myStepper2.step(10, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.MICROSTEP)






s.close()
