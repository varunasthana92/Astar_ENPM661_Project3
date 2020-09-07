# Astar Path Planner Phase 1

<p align="center">
	<img src="https://github.com/varunasthana92/Astar_Path_Planner/blob/master/Phase_1/sample%20outputs/output_rigid.gif" width="600">
</p>

## Overview

Phase 1 has one python scripts that generates a 2D map with obstacles and finds a path to travel from a user-defined start and end point. __SCRIPT REQUIRE PYTHON2.7 TO RUN. PYTHON3 WILL NOT WORK.__ Python 2.7 was used to test the script. Script is used for a rigid robot that has a defined radius and clearance. The code is organized in the following sections:

1. Libraries
2. Maps
3. Actions and Nodes
4. User Input
5. Exploration
6. Path Generation

### Dependencies
* matplotlib.pyplot
* cv2 (version 3.3)

### How to run
Phase 1 is the implementation of A* algorithm for a rigid robot with no holonomic constraint. Action set looks for the best move considering the angle in [0,360) with an incremental value of 30 degrees. Thus the computation speed is low for small step size. It is recommended to use a step size of 10 for quick results.<br><br>
Incorrect input data may terminate the program. In such case re-run with correct input (as per the instructions on the screen).

```
$ git clone https://github.com/varunasthana92/Astar_Path_Planner.git
$ cd Phase_1
$ python2 Astar_rigid.py
```
### Section 1: Libraries

The libraries imported for this project are:

* __numpy__ - used to handle the array data structures
* __copy__ - used to take copies of arrays into functions
* __math__ - used to generate fixed obstacles in the map
* __matplotlib.pyplot__ - used to draw the map and provide a visual
* __cv2__ - used to save the image as a JPEG
* __time__ - used to keep track of time when solving

### Section 2: Map

The final map is defined with five obstacles and hard-coded to specific dimensions. For a point robot, the size is fixed. For a rigid robot, the obstacles expand to account for the radius and clearance, and the program then treats the problem as a point robot.

* __Class Isobs()__ is used to check wheter a given float(x,y) coordinate is in free space or in the obstacle. This uses the concept of Half-Planes and hence the configuration space is treated as a continuoys space and is __not discretized__.
* __Class FinalMap()__ is used to generate the map and store it as an image for visualization. This required to discretize the configuration space into a matrix

### Section 3: Actions and Nodes

Contains functions and objects to define the action set and keep track of all nodes explored and visited.

### Section 4: User Input

Generates print statements while the program is running to collect user input data in the command line. When the rigid robot script is run, the user will be asked the following:

* Enter the radius and clearance (enter 0,0 for point robot) [separated by commas]: 
Ceil value of radius + clearance will be considered
* Enter the initial starting coordinates and theta [0,360) with origin at bottom left as x,y,t [separated by commas]: 
* Enter the step size "d" in integer (1<=d<=10):
* Enter the goal coordinates with origin at bottom left as x,y [separated by commas]: 

When giving the radius and clearance, the input is taken by the ceil value of radius + clearance. After giving the radius and clearance, a display map will pop up as a reference (__NOTE: The origin 0,0 is in the bottom-left. The y-axis shown is inverted than shown. Y starts at 0 in the bottom-left corner and goes to 200 in the top-left corner__). This helps the user choose starting and goal points in cartesian coodinates. If the user gives a point that is inside an obstacle or outside the map, the program will through an error and ask the user to re-enter the point.

### Section 5: Exploration

The code will display "Processing..." and begin searching the map from the starting location given by the user.

If the explored area becomes blocked from reaching the goal, the program will stop exploring and the program will end with the message "No solution exist, terminating...."

The run speed of the code is extremely slow with step soze of 1, while with step size above 5 the code has good run speed.

### Section 6: Path Generation

After the goal point is reached, a path will be traced back from the goal to the starting point. On the map, this path will be drawn in RED. In the file location, an image "back_tracking.png" is saved. At the end, the user can type any number and press enter to exit the program. This will close the image and the program.
