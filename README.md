# Autonomous-RC-Car

Our project on ‘Autonomous RC Car’ aims to convert a regular remote-controlled car into an intelligent system capable of navigating dynamic tracks in a slightly controlled environment. We are using sensors like Pi Camera and HC-SR04 Ultrasonic Range sensor interfaced through the Raspberry Pi Zero W mounted on the car to relay information about environment surrounding the car. This information is sent to the Laptop (processing unit) via Wi-Fi network. The laptop processes the image inputs through already trained neural networks, sends an output to the Arduino Uno board via the serial port. Arduino board signals the remote controller so that the car can move in the most suitable manner.

This video shows the autonomous navigation and collision avoidance capabilities of our finished project: https://www.youtube.com/watch?v=R1NxqouO4bs
