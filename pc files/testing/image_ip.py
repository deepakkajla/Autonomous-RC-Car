import numpy as np
import cv2
import pygame
from pygame.locals import *
import time
import os
import glob
import sys


class ImageIP(object):
    
    def __init__(self):
        # create labels
        self.m = np.zeros((4, 4), 'float')
        for i in range(4):
            self.m[i, i] = 1
        self.temp = np.zeros((1, 4), 'float')
        self.ImgCollection()

    def ImgCollection(self):
        saved = 0
        total = 0

        t1 = cv2.getTickCount()
        img_arr = np.zeros((1, 38400))
        label_arr = np.zeros((1, 4), 'float')
        #collected_img = glob.glob('img/*.jpg')

        print 'Image reading started...'

        # stream images 1 at a time
    
        frame = 1
        for jpg in os.listdir('img/'):
            image=cv2.imread(os.path.join('img/',jpg));
            if image is not None:
                # extracting bottom half of image as region of interest
                roi = image[120:240,:,0]
                #cv2.imshow('roi_image', roi)
                cv2.imshow('image', roi)
                # reshape the roi image into one row array
                temp = roi.reshape(1, 38400).astype(np.float32)             
                frame += 1
                total += 1
                key_pressed=cv2.waitKey()
                
                if key_pressed==ord('w'):
                    print("Forward")
                    saved += 1
                    img_arr = np.vstack((img_arr, temp))
                    label_arr = np.vstack((label_arr, self.m[0])) #forward as [1 0 0 0]
                    
                elif key_pressed==ord('s'):
                    print("Reverse")
                    saved += 1
                    img_arr = np.vstack((img_arr, temp))
                    label_arr = np.vstack((label_arr, self.m[1])) #reverse as [0 1 0 0]
                                                
                elif key_pressed==ord('d'):
                    print("Right")
                    saved += 1
                    img_arr = np.vstack((img_arr, temp))
                    label_arr = np.vstack((label_arr, self.m[2])) #right as [0 0 1 0]
                    
                elif key_pressed==ord('a'):
                    print("Left")
                    saved += 1
                    img_arr = np.vstack((img_arr, temp))
                    label_arr = np.vstack((label_arr, self.m[3])) #left as [0 0 0 1]
                    
                elif key_pressed==ord('q'):
                    print 'exiting...'
                    break

        # save images and labels
        train = img_arr[1:, :]
        train_labels = label_arr[1:, :]

        # save data as a numpy file
        file_name = str(int(time.time()))
        directory = "img"
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

if __name__ == '__main__':
    ImageIP()