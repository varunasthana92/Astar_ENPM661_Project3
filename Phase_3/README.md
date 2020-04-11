# Astar_Search_Algorithm_ENPM661-Project-3 Phase3

## Overview

Project 3 phase 3 has one python scripts that generates a 2D map with obstacles and finds a path to travel from a user-defined start and end point. __SCRIPT REQUIRE PYTHON2.7 TO RUN. PYTHON3 WILL NOT WORK.__ Python 2.7 was used to test the script. Script is used for a rigid robot that has a defined radius.

### Dependencies
* numpy
* math
* matplotlib.pyplot
* cv2 (version 3.3)
* time
* heapq
* argparse

### How to run
```
$ git clone https://github.com/varunasthana92/Astar_ENPM661_Project3.git
$ cd Astar_ENPM661_Project3/Phase_3
$ python2 astar.py
```
It also accpets an argument to display the explored nodes, --exp with default value as 0. If set to 1, explored nodes will be ploted in BLUE, but the process is time consuming.
```
$ python2 astar.py --exp=1
```

### User Inputs
All inputs are to be given in METERS
* Robot clearance (eg: 0.1)
* Robot initial position in x,y and theta (in degrees) with origin at the center of the map (eg: -4,-3,120)
* Left and right wheel RPM (eg: 90,90)
* Goal position in x,y (eg: 0,-3)

RPM is converted to the unit of "rad/sec" by multiplying the user input with 2pi/60.

Theta is measured anti-clockwise from positive x-axis.

Least count of 1cm is considered for the complete system with a threshold of 0.5cm in x and y coordinates and 30 degrees in rotation for node generation
Eg 1.014 m will be treated as 1.010 m or 101.0 cm
Eg 1.015 m will be treated as 1.015 m or 101.0 cm
Eg 1.015 m will be treated as 1.016 m or 102.0 cm

Also, ceil value of radius + clearance will be considered. Threshold for reaching the goal is set at 0.1m (or 10cm). 

### Path Generation
The run speed from top left to bottom right of the map (withut plotting of explored nodes) is around 5 mins.

After the goal point is reached, a path will be traced back from the start to goal point. On the map, this path will be drawn in RED. In the file location, an image "back_tracking.png" is saved. At the end, the user can type any number and press enter to exit the program.