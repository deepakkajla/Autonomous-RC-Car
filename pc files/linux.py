import threading
import SocketServer
import serial
import cv2
import numpy as np
import math
import time

# distance data measured by ultrasonic sensor
sensor_data = " "


class NeuralNetwork(object):

    def __init__(self):
        self.model = cv2.ml.ANN_MLP_create()
    
    def create(self):
        layer = np.int32([38400, 40, 4])
        self.model.setLayerSizes(layer)
        self.model = cv2.ml.ANN_MLP_load('mlp_trained/trained_network.xml')

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)


class RCControl(object):

    def __init__(self):
        self.serial_port = serial.Serial('/dev/ttyACM0', 115200, timeout=.1)

    def steer(self, prediction):
        if prediction == 0:
            self.serial_port.write(chr(1))
            print("Forward")
        elif prediction == 2:
            self.serial_port.write(chr(4))
            print("Right")
        elif prediction == 3:
            self.serial_port.write(chr(3))
            print("Left")
        else:
            self.stop()
        self.serial_port.write(chr(0))

    def stop(self):
        self.serial_port.write(chr(0))


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
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (255, 255, 255), 2)
            v = y_pos + height - 5

            # object name
            if name == 'STOP':
                cv2.putText(image, 'STOP', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif name == 'GREEN':
                cv2.putText(image, 'GREEN', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                self.green_light = True
            else:
                cv2.putText(image, 'RED', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                self.red_light = True
        
        return v


class SensorDataHandler(SocketServer.BaseRequestHandler):

    allow_reuse_address = True

    data = " "

    def handle(self):
        global sensor_data
        try:
            while self.data:
                self.data = self.request.recv(1024)
                sensor_data = round(float(self.data), 1)
                print sensor_data
        finally:
            print "Connection closed on thread 2"


class VideoStreamHandler(SocketServer.StreamRequestHandler):

    allow_reuse_address = True

    # height of camera
    h = 10  # cm

    # create neural network
    model = NeuralNetwork()
    model.create()

    obj_detection = ObjectDetection()
    rc_car = RCControl()

    # cascade classifiers
    stop_cascade = cv2.CascadeClassifier('haar_cascade/stop_sign.xml')
    red_cascade = cv2.CascadeClassifier('haar_cascade/red_lit.xml')
    green_cascade = cv2.CascadeClassifier('haar_cascade/green_lit.xml')

    d_to_camera = DistanceToCamera()
    d_stop_sign = 40
    d_glight = 40
    d_rlight = 40

    stop_start = 0              # start time when stop at the stop sign
    stop_finish = 0
    stop_time = 0
    drive_time_after_stop = 0

    def handle(self):

        global sensor_data
        stream_bytes = ' '
        stop_flag = False
        stop_sign_active = True
        ctr = -1

        # stream video frames one by one
        try:
            while True:
                ctr +=  1
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
                    
                    if ctr % 5 is 0:
                        # lower half of the image
                        half_gray = gray[120:240, :]
                        # reshape image
                        img_arr = half_gray.reshape(1, 38400).astype(np.float32)
                        # neural network makes prediction
                        prediction = self.model.predict(img_arr)

                    # stop conditions
                    if sensor_data is not None and sensor_data < 15:
                        print("Stopping, collision imminent")
                        self.rc_car.stop()
                    
                    elif 0 < self.d_stop_sign < 33 and stop_sign_active:
                        print("Stop sign ahead")
                        self.rc_car.stop()

                        # stop for 5 seconds
                        if stop_flag is False:
                            self.stop_start = cv2.getTickCount()
                            stop_flag = True
                        self.stop_finish = cv2.getTickCount()

                        self.stop_time = (self.stop_finish - self.stop_start)/cv2.getTickFrequency()
                        print "Stop time: {:.2f} s".format(self.stop_time)

                        # 5 seconds later, continue driving
                        if self.stop_time > 5:
                            print("Waited for 5 seconds")
                            stop_flag = False
                            stop_sign_active = False

                    elif 0 < self.d_glight < 33 and self.obj_detection.green_light:
                        print("Green light")
                        self.d_glight = 40
                        self.obj_detection.green_light = False
                        pass

                    elif 0 < self.d_rlight < 33 and self.obj_detection.red_light:
                        print("Red light")
                        self.rc_car.stop()
                        self.d_rlight = 40
                        self.obj_detection.red_light = False
                        pass

                    else:
                        if ctr % 5 is 0:
                            self.rc_car.steer(prediction)
                        self.stop_start = cv2.getTickCount()
                        self.d_stop_sign = 40

                        if stop_sign_active is False:
                            self.drive_time_after_stop = (self.stop_start - self.stop_finish)/cv2.getTickFrequency()
                            if self.drive_time_after_stop > 5:
                                stop_sign_active = True

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.rc_car.stop()
                        break

            cv2.destroyAllWindows()

        finally:
            print "Connection closed on thread 1"


class ThreadServer(object):

    def server_thread1(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.allow_reuse_address = True
        server.serve_forever()

    def server_thread2(host, port):
        server = SocketServer.TCPServer((host, port), SensorDataHandler)
        server.allow_reuse_address = True
        server.serve_forever()

    vport = raw_input("Enter video stream port:")
    uport = raw_input("Enter u-sonic stream port:")
    video_thread = threading.Thread(target=server_thread1,args=('192.168.43.152', int(vport)))
    video_thread.start()
    distance_thread = threading.Thread(target=server_thread2,args=('192.168.43.152', int(uport)))
    distance_thread.start()

if __name__ == '__main__':
    ThreadServer()
