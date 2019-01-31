import threading
import SocketServer
import serial
import cv2
import numpy as np
import math

class DistanceToCamera(object):

    def __init__(self):
        # picam parameters from calibration
        self.alpha = 8.0 * math.pi / 180 #tilt angle of camera
        self.v0 = 113.593399626
        self.ay = 317.826404563

    def calculate(self, v, h, x_shift, image):
        # compute and return the distance from the target point to the camera
        d = h / math.tan(self.alpha + math.atan((v - self.v0) / self.ay))
        if d > 0:
            cv2.putText(image, "%.1fcm" % d,
                (image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return d


class ObjectDetection(object):
    def __init__(self):
        self.red_light = False
        self.green_light = False
    
    def detect(self, cascade_classifier, name, gray_image, image):
        # y camera coordinate of the target point 'P'
        v = 0

        # detection
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (255, 255, 255), 2)
            v = y_pos + height - 5

            # object name
            cv2.putText(image, name, (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return v


class VideoStreamHandler(SocketServer.StreamRequestHandler):

    # height of camera
    h = 10  # cm

    obj_detection = ObjectDetection()
    #rc_car = RCControl()

    # cascade classifiers
    stop_cascade = cv2.CascadeClassifier('E:/8th Sem/Capstone/Self Driving Car/pc files/haar_cascade/stop_sign.xml')
    red_cascade = cv2.CascadeClassifier('E:/8th Sem/Capstone/Self Driving Car/pc files/haar_cascade/red_lit.xml')
    green_cascade = cv2.CascadeClassifier('E:/8th Sem/Capstone/Self Driving Car/pc files/haar_cascade/green_lit.xml')

    d_to_camera = DistanceToCamera()
    d_stop_sign = 25
    d_glight = 25
    d_rlight = 25

    stop_start = 0              # start time when stop at the stop sign
    stop_finish = 0
    stop_time = 0
    drive_time_after_stop = 0

    def handle(self):

        #global sensor_data
        stream_bytes = ' '
        stop_flag = False
        stop_sign_active = True

        # stream video frames one by one
        try:
            while True:
                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), 0)
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), -1)

                    # object detection
                    v_param1 = self.obj_detection.detect(self.stop_cascade, 'STOP', gray, image)
                    v_param2 = self.obj_detection.detect(self.red_cascade, 'RED', gray, image)
                    v_param3 = self.obj_detection.detect(self.green_cascade, 'GREEN', gray, image)

                    # distance measurement
                    if v_param1 > 0 or v_param2 > 0 or v_param3 > 0:
                        d1 = self.d_to_camera.calculate(v_param1, self.h, 300, image)
                        d2 = self.d_to_camera.calculate(v_param2, self.h, 300, image)
                        d3 = self.d_to_camera.calculate(v_param2, self.h, 300, image)
                        self.d_stop_sign = d1
                        self.d_rlight = d2
                        self.d_glight = d3

                    cv2.imshow('image', image)
                    #cv2.imshow('mlp_image', half_gray)

                    if 0 < self.d_stop_sign < 25 and stop_sign_active:
                        print("Stop sign ahead")

                        # stop for 5 seconds
                        if stop_flag is False:
                            self.stop_start = cv2.getTickCount()
                            stop_flag = True
                        self.stop_finish = cv2.getTickCount()

                        self.stop_time = (self.stop_finish - self.stop_start)/cv2.getTickFrequency()
                        print "Stop time: %.2fs" % self.stop_time

                        # 5 seconds later, continue driving
                        if self.stop_time > 5:
                            print("Waited for 5 seconds")
                            stop_flag = False
                            stop_sign_active = False

                    elif 0 < self.d_rlight < 25:
                        print("Red light")
                        
                    elif 0 < self.d_glight < 25:
                        print("Green light")
                        
                    else:
                        self.stop_start = cv2.getTickCount()
                        self.d_stop_sign = 25

                        if stop_sign_active is False:
                            self.drive_time_after_stop = (self.stop_start - self.stop_finish)/cv2.getTickFrequency()
                            if self.drive_time_after_stop > 5:
                                stop_sign_active = True

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cv2.destroyAllWindows()

        finally:
            print "Connection closed on thread 1"


class ThreadServer(object):

    def server_thread(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    video_thread = threading.Thread(target=server_thread('192.168.43.19', 8000))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()
