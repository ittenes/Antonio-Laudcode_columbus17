import cv2

vidcap = cv2.VideoCapture()
vidcap.open(1)
retval, image = vidcap.retrieve()
vidcap.release()

cv2.imwrite("test.png", image)
