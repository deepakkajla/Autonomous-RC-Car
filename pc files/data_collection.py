import numpy as np
import cv2
import serial
import pygame
from pygame.locals import *
import socket
import time
import os


class DataCollection(object):
    
    def __init__(self):
       
        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.43.19', 8000))
        self.server_socket.listen(0)

        # establish connection
        self.connection = self.server_socket.accept()[0].makefile('rb')

        pygame.init()
        pygame.display.set_mode([400, 300])
        # connect to serial port
        self.ser = serial.Serial('COM3', 115200, timeout=.1)
        time.sleep(1)
        self.send_inst = True
        
        # create labels
        self.m = np.zeros((4, 4), 'float')
        for i in range(4):
            self.m[i, i] = 1
        self.temp = np.zeros((1, 4), 'float')

        self.ImgCollection()
        pygame.display.update()

    def ImgCollection(self):

        saved = 0
        total = 0

        t1 = cv2.getTickCount()
        img_arr = np.zeros((1, 38400))
        label_arr = np.zeros((1, 4), 'float')

        print 'Image collection started...'

        # stream images 1 frame at a time
        try:
            stream_bytes = ' '
            frame = 1
            while self.send_inst:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), 0) #grayscale image
                    
                    # extracting bottom half of image as region of interest
                    roi = image[120:240, :]
                    
                    # save streamed images
                    cv2.imwrite('collection_images/img{:>05}.jpg'.format(frame), image)
                    
                    #cv2.imshow('roi_image', roi)
                    cv2.imshow('image', image)
                    
                    # reshape the roi image into one row array
                    temp = roi.reshape(1, 38400).astype(np.float32)
                    
                    frame += 1
                    total += 1

                    # get input from human driver
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            key_pressed = pygame.key.get_pressed()
                            
                            self.ser.flush()

                            if key_pressed[pygame.K_UP]:
                                print("Forward")
                                self.ser.write(chr(1))
                                saved += 1
                                img_arr = np.vstack((img_arr, temp))
                                label_arr = np.vstack((label_arr, self.m[0])) #forward as [1 0 0 0]
                                
                            elif key_pressed[pygame.K_DOWN]:
                                print("Reverse")
                                self.ser.write(chr(2))
                                saved += 1
                                img_arr = np.vstack((img_arr, temp))
                                label_arr = np.vstack((label_arr, self.m[1])) #reverse as [0 1 0 0]
                                                            
                            elif key_pressed[pygame.K_RIGHT]:
                                print("Right")
                                self.ser.write(chr(4))
                                saved += 1
                                img_arr = np.vstack((img_arr, temp))
                                label_arr = np.vstack((label_arr, self.m[2])) #right as [0 0 1 0]
                                
                            elif key_pressed[pygame.K_LEFT]:
                                print("Left")
                                self.ser.write(chr(3))
                                saved += 1
                                img_arr = np.vstack((img_arr, temp))
                                label_arr = np.vstack((label_arr, self.m[3])) #left as [0 0 0 1]
                                
                            elif key_pressed[pygame.K_x] or key_pressed[pygame.K_q]:
                                print 'exiting...'
                                self.send_inst = False
                                self.ser.write(chr(0))
                                self.ser.close()
                                break
                                    
                        elif event.type == pygame.KEYUP:
                            self.ser.write(chr(0))

            # save images and labels
            train = img_arr[1:, :]
            train_labels = label_arr[1:, :]

            # save data as a numpy file
            file_name = str(int(time.time()))
            directory = "collection_data"
            if not os.path.exists(directory):
                os.makedirs(directory)
            try:    
                np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
            except IOError as e:
                print(e)

            t2 = cv2.getTickCount()
            # calculate streaming duration
            timed = (t2 - t1) / cv2.getTickFrequency()
            print 'Collection time:', timed

            print(train.shape)
            print(train_labels.shape)
            print 'Total frames:', total
            print 'Saved frames:', saved
            print 'Lost frames:', total - saved

        finally:
            self.connection.close()
            self.server_socket.close()

if __name__ == '__main__':
    DataCollection()
