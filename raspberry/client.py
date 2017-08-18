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

# CONEXION

TCP_IP = '192.168.1.36'
TCP_PORT = 5001
s = socket.socket()
s.connect((TCP_IP, TCP_PORT))


# MOTOR
''' CALIBRATE SISTEM
    motor 01 in GPIO 19 - 24
    motor 02 in GPIO 21 - 26
    you can calibrate the system with buttons'''

mh = Adafruit_MotorHAT()

# recommended for auto-disabling motors on shutdown!


def turnOffMotors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)


atexit.register(turnOffMotors)

myStepper1 = mh.getStepper(200, 1)  # 200 steps/rev, motor port #1
myStepper1.setSpeed(30)             # 30 RPM

myStepper2 = mh.getStepper(200, 2)  # 200 steps/rev, motor port #2
myStepper2.setSpeed(300)             # 30 RPM

# create empty threads (these will hold the stepper 1 and 2 threads)
st1 = threading.Thread()
st2 = threading.Thread()

stepstyles = [Adafruit_MotorHAT.SINGLE, Adafruit_MotorHAT.DOUBLE,
              Adafruit_MotorHAT.INTERLEAVE, Adafruit_MotorHAT.MICROSTEP]
#--------------------------------------------

IO.setwarnings(False)           # do not show any warnings
#                               # integer for storing the delay multiple
IO.setmode(IO.BCM)
# motor 01 FORWARD
IO.setup(21, IO.IN)              # initialize GPIO21 as an input.
# motor 01 BACKWARD
IO.setup(20, IO.IN)              # initialize GPIO22 as an input.
# motor 02 FORWARD
IO.setup(19, IO.IN)              # initialize GPIO19 as an input.
# motor 02 BACKWARD
IO.setup(16, IO.IN)              # initialize GPIO16 as an input.


def stepper_worker(stepper, numsteps, direction, style):
    print("Steppin!")
    stepper.step(numsteps, direction, style)
    print("Done")


step_1 = 0
step_2 = 0

length = None
buffer = ""

w = 1024.0
h = 768.0
capture = cv2.VideoCapture(0)
capture.set(3, w)
capture.set(4, h)

x1 = 1
x2 = 1
try:
    while 1:

        # FOTO
        ret, frame = capture.read()

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = numpy.array(imgencode)
        stringData = data.tostring()

        decimg = cv2.imdecode(data, 1)

        #cv2.namedWindow('imageWindow', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('imageWindow',decimg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows('imageWindow')

        s.send(str(len(stringData)).ljust(16))
        s.send(stringData)

        # DATOS
        data_pos = s.recv(50)
        array_data = data_pos.split('_')

        x1 = array_data[0]
        x2 = array_data[2]
        y1 = array_data[1]
        y2 = array_data[3]

        print ("POSITION:"), x1, x2, y1, y2

        ''' POSITION OF DE PIGNION EYE AND AIM THE LASER '''

        # CALIBRAR MOTOR 01
        for y in range(x1):
            time.sleep(0.01)
            if(IO.input(21) == False):  # if button1 is pressed
                #myStepper1.step(1, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.MICROSTEP)
                if(x1 < 100):
                    x1 = x1 + 1  # increment x by one if x<100
                    st1F = threading.Thread(
                        target=stepper_worker,
                        args=(myStepper1, 1, Adafruit_MotorHAT.FORWARD,
                              stepstyles[0],)
                    )
                    st1F.start()
                    time.sleep(0.01)
                    step_1 += 1
                    print step_1
            if(IO.input(20) == False):  # if button1 is pressed
                #myStepper1.step(1, Adafruit_MotorHAT.BACKWARD,  Adafruit_MotorHAT.MICROSTEP)
                if(x1 > 1):
                    x1 = x1 - 1
                    st1B = threading.Thread(
                        target=stepper_worker,
                        args=(myStepper1, 1, Adafruit_MotorHAT.BACKWARD,
                              stepstyles[0],)
                    )
                    st1B.start()
                    time.sleep(0.01)
                    step_1 -= 1
                    print step_1

        # CALIBRAR MOTOR 02
        for y in range(x2):
            time.sleep(0.01)
            if(IO.input(19) == False):  # if button1 is pressed
                #myStepper2.step(1, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.MICROSTEP)
                if(x2 < 100):
                    x2 = x2 + 1
                    st2F = threading.Thread(
                        target=stepper_worker,
                        args=(myStepper2, 1, Adafruit_MotorHAT.FORWARD,
                              stepstyles[0],)
                    )
                    st2F.start()
                    time.sleep(0.01)
                    step_2 += 1
                    print step_2
            if(IO.input(16) == False):  # if button1 is pressed
                #myStepper2.step(1, Adafruit_MotorHAT.BACKWARD,  Adafruit_MotorHAT.MICROSTEP)
                if(x2 > 1):
                    x2 = x2 - 1
                    st2B = threading.Thread(
                        target=stepper_worker,
                        args=(myStepper2, 1, Adafruit_MotorHAT.BACKWARD,
                              stepstyles[0],)
                    )
                    st2B.start()
                    time.sleep(0.01)
                    step_2 -= 1
                    print step_2

        ''' END CALIBRATE SISTEM
        ------------------------------------------------------'''


except KeyboardInterrupt:
    IO.cleanup()
    s.close()

IO.cleanup()

s.close()
