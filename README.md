# Astar Path Planner for rigid robot

## By Varun Asthana

### University of Maryland

## Overview

This project has 3 sub projects- Phase 1, Phase 2 and Phase 3.  
Initial position has a parameter of __theta__ defining the heading angle in the range [0, 360). Solver looks for a path with defined step size and action set.

* Phase 1 is the implementation of A* algorithm for a rigid robot with no holonomic constraint.

<p align="center">
	<img src="https://github.com/varunasthana92/Astar_Path_Planner/blob/master/Phase_1/sample%20outputs/output_rigid.gif" width="600">
</p>

* Phase 2 is the implementation of A* algorithm for a rigid robot with non-holonomic constraints of a differential drive.

* Phase 3 is the simulation of Phase 2 for a TurtleBot in Gazebo.
<p align="center">
	<img src="https://github.com/varunasthana92/Astar_Path_Planner/blob/master/Phase_1/sample%20outputs/demo1_speed8x.gif" width="600">
</p>

## Download
```
$ git clone https://github.com/varunasthana92/Astar_Path_Planner.git
```