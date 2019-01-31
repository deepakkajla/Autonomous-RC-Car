import serial
import pygame
import time
from pygame.locals import *


class RCTest(object):

    def __init__(self):
        pygame.init()
        pygame.display.set_mode([400, 300])
        self.ser = serial.Serial('COM3', 115200, timeout=.1)
        time.sleep(1)
        self.send_inst = True
        self.steer()
        pygame.display.update()

    def steer(self):

        while self.send_inst:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    key_pressed = pygame.key.get_pressed()

                    self.ser.flush()

                    # simple orders
                    if key_pressed[pygame.K_UP]:
                        print("Forward")
                        self.ser.write(chr(1))

                    elif key_pressed[pygame.K_DOWN]:
                        print("Reverse")
                        self.ser.write(chr(2))

                    elif key_pressed[pygame.K_RIGHT]:
                        print("Right")
                        self.ser.write(chr(4))

                    elif key_pressed[pygame.K_LEFT]:
                        print("Left")
                        self.ser.write(chr(3))

                    # exit
                    elif key_pressed[pygame.K_x] or key_pressed[pygame.K_q]:
                        print 'Exit'
                        self.send_inst = False
                        self.ser.write(chr(0))
                        self.ser.close()
                        break

                elif event.type == pygame.KEYUP:
                    self.ser.write(chr(0))

if __name__ == '__main__':
    RCTest()
