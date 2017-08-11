#!/usr/bin/python
import socket
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor, Adafruit_StepperMotor
import time
import atexit
import threading
import random
import RPi.GPIO as IO
import sys
import cv2
import numpy

TCP_IP = '192.168.1.36'
TCP_PORT = 5001
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)

''' CALIBRATE SISTEM
    motor 01 in GPIO 19 - 24
    motor 02 in GPIO 21 - 26
    you can calibrate the system with buttons'''

######## PARTE PARA QUITAR CUANDO SE INTEGRE

mh = Adafruit_MotorHAT()

# recommended for auto-disabling motors on shutdown!
def turnOffMotors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)

atexit.register(turnOffMotors)

myStepper1 = mh.getStepper(400, 1)  # 200 steps/rev, motor port #1
myStepper1.setSpeed(255)             # 30 RPM
myStepper2 = mh.getStepper(400, 2)  # 200 steps/rev, motor port #2
myStepper2.setSpeed(255)             # 30 RPM

# create empty threads (these will hold the stepper 1 and 2 threads)
st1 = threading.Thread()
st2 = threading.Thread()

stepstyles = [Adafruit_MotorHAT.SINGLE, Adafruit_MotorHAT.DOUBLE, Adafruit_MotorHAT.INTERLEAVE, Adafruit_MotorHAT.MICROSTEP]
#--------------------------------------------

IO.setwarnings(False)           # do not show any warnings
#                               # integer for storing the delay multiple
IO.setmode (IO.BCM)

# motor 01 FORWARD
IO.setup(21,IO.IN)              # initialize GPIO21 as an input.
# motor 01 BACKWARD
IO.setup(20,IO.IN)              # initialize GPIO22 as an input.
# motor 02 FORWARD
IO.setup(19,IO.IN)              # initialize GPIO19 as an input.
# motor 02 BACKWARD
IO.setup(16,IO.IN)              # initialize GPIO16 as an input.

def stepper_worker(stepper, numsteps, direction, style):
    print("Steppin!")
    stepper.step(numsteps, direction, style)
    print("Done")

step_1 = 0
step_2 = 0         # 30 RPM

length = None
buffer = ""

w = 1280
h = 1024
capture = cv2.VideoCapture(0)
capture.set(3,w)
capture.set(4,h)

try:
    while (True):

        #FOTO

        ret, frame = capture.read()

        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = numpy.array(imgencode)
        stringData = data.tostring()

        decimg=cv2.imdecode(data,1)
        cv2.imshow('CLIENT',decimg)

        sock.send( str(len(stringData)).ljust(16))
        sock.send( stringData )

        #DATOS
        while:
            try:
                data_pos = s.recv(50)
                buffer += data_pos
                length_str, ignored, buffer = buffer.rpartition(':')
                buffer, ignored, length = length_str.rpartition(':')
                print("Microsteps", length)
                myStepper1.step(10, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.MICROSTEP)
                break
            except socket.error, e:
                time.sleep(1)    # Something else happened, handle error, exit, etc.
                print e


        if(IO.input(21) == False):            #if button1 is pressed
            #myStepper1.step(1, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.MICROSTEP)
            st1F = threading.Thread(target=stepper_worker, args=(myStepper1, 1, Adafruit_MotorHAT.FORWARD, stepstyles[3],))
            st1F.start()
            time.sleep(0.01)
            step_1 += 1
            print step_1
        if(IO.input(20) == False):            #if button1 is pressed
            #myStepper1.step(1, Adafruit_MotorHAT.BACKWARD,  Adafruit_MotorHAT.MICROSTEP)
            st1B = threading.Thread(target=stepper_worker, args=(myStepper1, 1, Adafruit_MotorHAT.BACKWARD, stepstyles[2],))
            st1B.start()
            time.sleep(0.01)
            step_1 -= 1
            print step_1
        if(IO.input(19) == False):            #if button1 is pressed
            #myStepper2.step(1, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.MICROSTEP)
            st2F = threading.Thread(target=stepper_worker, args=(myStepper2, 1, Adafruit_MotorHAT.FORWARD, stepstyles[2],))
            st2F.start()
            time.sleep(0.01)
            step_2 += 1
            print step_2
        if(IO.input(16) == False):            #if button1 is pressed
            #myStepper2.step(1, Adafruit_MotorHAT.BACKWARD,  Adafruit_MotorHAT.MICROSTEP)
            st2B = threading.Thread(target=stepper_worker, args=(myStepper2, 1, Adafruit_MotorHAT.BACKWARD, stepstyles[2],))
            st2B.start()
            time.sleep(0.01)
            step_2 -= 1
            print step_2

        ''' END CALIBRATE SISTEM
        ------------------------------------------------------'''

        ''' POSITION OF DE PIGNION EYE AND AIM THE LASER '''

        data = ('100_200_300_400')

        array_data = data.split('_')

        x1 = array_data[0]
        x2 = array_data[2]
        y1 = array_data[1]
        y2 = array_data[3]

        print x1, x2, y1, y2

        w = 1280
        h = 1024



except KeyboardInterrupt:
    IO.cleanup()
    s.close()

IO.cleanup()

s.close()
