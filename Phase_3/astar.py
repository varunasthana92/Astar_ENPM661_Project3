# ENPM 661 PROJECT 3 PHASE 2
# Varun Asthana

# =====SECTION 1: LIBRARIES=====
import numpy as np
import copy as cp
import math
import matplotlib.pyplot as plt
import sys
# sys.path.remove(sys.path[1])
import cv2
import time
from heapq import heappush, heappop
import argparse


# =====SECTION 2: MAPS=====
thres_x, thres_y, thres_t = 1, 1, 30.0
# Final fixed map that has five obstacles
class FinalMap():
    def __init__(self, height, width, clr, fact):
        """Initializes final map
        height:     row dimension [pixels]
        width:      column dimension [pixels]
        c:        clearance from map border"""
        height = height*fact
        width = width*fact
        clr = clr*fact
        clr = int(math.ceil(clr))
        self.c = clr
        self.f = fact
        self.grid= np.ones([height,width,3], dtype ='uint8')*255
        self.grid[0:(clr+1),:,0] = 0
        self.grid[height-(clr+1):height,:,0] = 0
        self.grid[:,0:(clr+1),0] = 0
        self.grid[:, width-(clr+1):width,0] = 0


    # Obstacle in top right
    def circ(self, radius, h, w):
        """Customizable circle obstacle
            radius:     radius dimention [pixels]
            h:          circle center location in map's coordinate system
            w:          circle center location in map's coordinate system"""
        h = h*self.f
        w = w*self.f
        radius = radius*self.f
        finalRad = radius + self.c
        if(h-finalRad<0):
            ha=0;
        else:
            ha= h - finalRad

        if(h+finalRad >= self.grid.shape[0]):
            hb= self.grid.shape[0]
        else:
            hb= h + finalRad

        if(w-finalRad<0):
            wa=0;
        else:
            wa= w - finalRad

        if(w+finalRad >= self.grid.shape[1]):
            wb= self.grid.shape[1]
        else:
            wb= w + finalRad

        for h_ in range(ha, hb):
            for w_ in range(wa, wb):
                eqn= (h_-h)**2 + (w_-w)**2
                if(eqn<=(finalRad**2)):
                    self.grid[h_,w_,0] = 0
        return

    # Obstacle in bottom left
    def sqr(self, side, h, w):
        """Fixed polygon shape for obstacle in bottom-left of map"""
        side = side*self.f
        h = h*self.f
        w = w*self.f
        hlen = side/2
        h1, w1 = h - hlen, w - hlen # top left
        h2, w2 = h - hlen, w + hlen # top right
        h3, w3 = h + hlen, w + hlen # bottom right
        h4, w4 = h + hlen, w - hlen # bottom left

        l1y = h1 - self.c
        l2x = w2 + self.c
        l3y = h3 + self.c
        l4x = w4 - self.c

        for x in range(self.grid.shape[1]):
            for y in range (self.grid.shape[0]):
                if(y>=l1y and y<=l3y and x<=l2x and x>=l4x):
                    self.grid[y,x, 0]=0
        # Highlighting the rectangle vertices
        self.grid[int(h1),int(w1),0:2]= 0
        self.grid[int(h2),int(w2),0:2]= 0
        self.grid[int(h3),int(w3),0:2]= 0
        self.grid[int(h4),int(w4),0:2]= 0
        return

# Class to have map representation as math equations and methods to check if a point
# is in free space or in the obstacle
class Obs():
    def __init__(self, height, width, clr, fact):
        """Initializes final map
        ht:     row dimension [pixels]
        wd:      column dimension [pixels]
        c:        clearance from map border"""
        self.ht = height * fact
        self.wd = width * fact
        clr = clr*fact
        clr = int(math.ceil(clr))
        self.c = clr
        self.f = fact

    def bound(self, x,y):
        # print('x',x)
        # print('y',y)
        if((y<=self.c) or y>=(self.ht- self.c)):
            return False
        if((x<=self.c) or x>=(self.wd-self.c)):
            return False
        return True


    def circ(self, radius, h, w, x, y):
        h = h*self.f
        w = w*self.f
        radius = radius*self.f
        finalRad = radius + self.c
        eqn= (y-h)**2 + (x-w)**2
        if(eqn<=(finalRad**2)):
            return False
        return True

    def sqr(self,side, h, w, x, y):
        side = side*self.f
        h = h*self.f
        w = w*self.f
        hlen = side/2
        h1, w1 = h - hlen, w - hlen # top left
        h2, w2 = h - hlen, w + hlen # top right
        h3, w3 = h + hlen, w + hlen # bottom right
        h4, w4 = h + hlen, w - hlen # bottom left

        l1y = h1 - self.c
        l2x = w2 + self.c
        l3y = h3 + self.c
        l4x = w4 - self.c

        if(y>=l1y and y<=l3y and x<=l2x and x>=l4x):
            return False
        return True

    def notObs(self, y, x):
        y = y*self.f
        x = x*self.f
        if(self.bound(x,y) and self.circ(1, 5, 5, x,y) and self.circ(1, 2, 7, x,y) and self.circ(1, 8, 3, x,y)):
            if(self.circ(1, 8, 7, x,y) and self.sqr(1.5, (1.25 + 1.5/2), (2.25+ 1.5/2),x,y)):
                if(self.sqr(1.5, 5, 1,x,y) and self.sqr(1.5, 5, 9,x,y)):
                    if(self.sqr(1.5, (1.25 + 1.5/2), (2.25+ 1.5/2),x,y)):
                        return True

        return False

# =====SECTION 3: ACTIONS AND NODES=====
class AllNodes():
    # Initialize class object
    def __init__(self, height, width, depth):
        """Initializes to keep track of all nodes explored"""
        self.h_ = height 
        self.w_ = width
        self.d_ = depth
        self.allStates=[]
        self.visited= np.zeros([self.h_, self.w_, self.d_])
        self.ownIDarr= np.ones([self.h_, self.w_, self.d_], dtype='int64')*(-1)
        self.pIDarr= np.ones([self.h_, self.w_, self.d_], dtype='int64')*(-1)
        self.cost2come= np.ones([self.h_, self.w_, self.d_], dtype=np.float64)*(float('inf'))
        # self.cost2go= np.zeros([self.h_, self.w_], dtype='f')
        # self.totCost= np.zeros([self.h_, self.w_, self.d_], dtype='f')
        self.actDone= np.ones([self.h_, self.w_, self.d_], dtype='int8')*(-1)
        self.sort=[]
    # Function to mark the node as visited in the visited array
#     def setCost2go(self, goal):
# #         print('goal', goal)
#         h = np.arange(0.,self.h_/2,0.5)
#         w = np.arange(0.,self.w_/2,0.5)
#         hh = (np.tile(h, (self.w_,1)).T).reshape(-1,1)
#         ww = np.tile(w, (self.h_,1)).reshape(-1,1)
# #         hh = hh.reshape(-1,1)
# #         ww = ww.reshape(-1,1)
#         hh = hh - goal[0]
#         ww = ww - goal[1]
#         cost = np.sqrt((hh**2) + (ww**2))
#         cost = cost.reshape(self.h_, self.w_)
#         self.cost2go = cost
# #         for h in range(self.h_):
# #             for w in range(self.w_):
# #                 dis= math.sqrt((goal[0]-h)**2 + (goal[1]-w)**2)
# #                 self.cost2go[h,w] = dis
#         return

    # Function to initialize total cost as sum of cost2go and cost2come
    # def setTotCost(self):
    #     for i in range (self.d_):
    #         self.totCost[:,:,i] = self.cost2come[:,:,i] + self.cost2go


    # # Function to mark the node as visited in the visited array
    # def updateVisited(self, cord):
    #     self.visited[cord[0], cord[1]] = 1
    #     return

    # Function to get update cost in cost2come array
    def updateCost2Come(self, cord, cost, pid, actId, currState, goalState):
        # print('array cost of tempNode', self.cost2come[cord[0], cord[1], cord[2]])
        if(self.cost2come[cord[0], cord[1], cord[2]] > cost):
            self.cost2come[cord[0], cord[1], cord[2]] = cost
            # self.totCost[cord[0], cord[1], cord[2]] = cost + self.cost2go[cord[0], cord[1]]
            # self.totCost[cord[0], cord[1], cord[2]] = cost + heu(curr,goal)
            c2g = heu(currState,goalState)
            totCost = cost + c2g
            # print('final state= ', finalState)
            # print('temp state= ', currState)
            # print('temp c2c =', cost)
            # print('temp c2g =', c2g)
            # print('totCost = ', totCost)
            self.pIDarr[cord[0], cord[1], cord[2]] = pid
            self.actDone[cord[0], cord[1], cord[2]] = actId
            data = [totCost, currState[0], currState[1], currState[2], pid, cost]
            # print("----")
            # print(data)
            heappush(self.sort, data)
        return

    # Function to add new unique node in the Nodes data set
    def visit(self, node, state):
        ownId = int(len(self.allStates))
        self.ownIDarr[node[0], node[1], node[2]] = ownId
        # self.visited[node[0], node[1]] = 1

        self.visited[node[0], node[1], node[2]] = 1

        # if(int(len(self.allStates)) == 34):
        #     print('own id 34 data')
        #     print('state', state)
        #     print('node', node)
        #     input()
        self.allStates.append(state)
        return ownId

    # Function to get own id
    def getOwnId(self,cord):
        return self.ownIDarr[cord[0], cord[1], cord[2]]

    # Function to get parent id
    def getParentId(self,cord):
        return self.pIDarr[cord[0], cord[1], cord[2]]

    # Function to get state of the node i.e. coordinate [h,w[]
    def getStates(self, idx):
        return self.allStates[idx]

    # Function to get the index value of total cost array having minimum cost value
    def minCostIdx(self, act, fact, obsChk, wr, wL):
        # print 'minimum'   
#         try:

        # newMin= np.min(self.totCost[self.totCost>0])

        # data = [self.totCost[cord[0], cord[1], cord[2]], curr[0], curr[1], curr[2], pid]
        data = heappop(self.sort)
        newMin = data[0]
#         print('in try')
        if(newMin == float('inf')):
            status= False
#             print('found inf', status)
            new_parentState = -1
            p_state = -1
        else:
# newMin= np.min(graph.totCost[graph.totCost>0])
# status= True
# index= np.argwhere(graph.totCost==newMin)[0]
# pid = graph.pIDarr[index[0],index[1],index[2]]
# p_state = graph.getStates(pid)
# p_scaled = scaled(p_state, fact)
# p_node = threshold_state(p_scaled)
# actId = graph.actDone[index[0], index[1], index[2]]


            status= True
            # index= np.argwhere(self.totCost==newMin)[0]
            # pid = self.pIDarr[index[0],index[1],index[2]]
            # p_state = self.getStates(pid)
            # p_scaled = scaled(p_state, fact)
            # p_node = threshold_state(p_scaled)
            # actId = self.actDone[index[0], index[1], index[2]]

            # _, new_parentState,_ = move(p_state, wr, wL, act[actId][0], act[actId][1],obsChk, False, True) # move(currState, r, L, UL, UR, obsChk, chk = True, plot= False):



            # data = [self.totCost[cord[0], cord[1], cord[2]], curr[0], curr[1], curr[2], pid]
            new_parentState = np.array([data[1], data[2], data[3]])
            pid = data[4]

#         except:
#             newMin = float ('inf')
#             status= False
#             print('except in func', status)
#             new_parentState = -1
#             p_state = -1
        return status, newMin, new_parentState, pid


def heu(current, goal): #defining heuristic function as euclidian distance between current node and goal
    h = math.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)
    # h = math.sqrt((current[0] - self.goal[0])**2 + (current[1] - self.goal[1])**2 + (current[2] - self.goal[2])**2)
    return h

# Function to convert float point states to respective regions in node format
def threshold_state(scaledState, tt=thres_t, tx=thres_x, ty=thres_y):
    node= np.ones(3, dtype='f')*(-1)
    if(scaledState[0] - int(scaledState[0]) < ty):
        node[0] = int(scaledState[0])
    elif(scaledState[0] - int(scaledState[0]) >= ty and scaledState[0] - int(scaledState[0]) <= (1.5*ty)):
        node[0] = int(scaledState[0]) + ty
    else:
        node[0] = int(scaledState[0]) + 1

    if(scaledState[1] - int(scaledState[1]) < tx):
        node[1] = int(scaledState[1])
    elif(scaledState[1] - int(scaledState[1]) >= tx and scaledState[1] - int(scaledState[1]) <= (1.5*tx)):
        node[1] = int(scaledState[1]) + tx
    else:
        node[1] = int(scaledState[1]) + 1

    if(scaledState[2] - (tt*(scaledState[2]//tt)) >= (tt/2)):
        node[2] = (tt*(scaledState[2]//tt)) + tt
    else:
        node[2] = (tt*(scaledState[2]//tt))
    if(node[2] == 360.):
        node[2] = 0
    node[0]= node[0]/ty
    node[1]= node[1]/tx
    node[2]= node[2]/tt
    return np.array(node, dtype='int32')

# Function to check if the goal node is reached
def goalReached(curr, goal, fact):
    rad = 0.1
    dis = np.sqrt((curr[0] - goal[0])**2 + (curr[1] - goal[1])**2)
    if (dis<= rad):
        # print('rad ', rad)
        # print('distance final', dis)
        return 1
    else:
        return 0

def scaled(state, fact, tt=thres_t):
    if(state[2] - (tt*(state[2]//tt)) >= (tt/2)):
        theta = (tt*(state[2]//tt)) + tt
    else:
        theta = (tt*(state[2]//tt))
    if(theta == 360.):
        theta = 0

    theta= int(theta/tt)


    return np.array([int(np.round(state[0]*fact)), int(np.round(state[1]*fact)), theta])

# def plotStates(src, dst):
#     X0 = src[1]
#     Y0 = src[0]
#     U0 = dst[1]-X0
#     V0 = dst[0]-Y0
#     plt.quiver(X0, Y0, U0, -V0, units='xy' ,scale=1,headwidth = 1, headlength=0)
#     return

# def plotBack(src, dst):
#     X0 = src[1]
#     Y0 = src[0]
#     U0 = dst[1]-X0
#     V0 = dst[0]-Y0
#     plt.quiver(X0, Y0, U0, -V0, units='xy' , scale=1, color= 'r', headwidth = 1, headlength=0)
#     return

# Function to take a step in the direction of speficied angle degree
# def action(step, degree):
#     if(degree >=360):
#         degree = degree-360
#     t= math.radians(degree)
#     rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype= 'f')
#     base_move = np.array([step,0], dtype= 'f')
#     move_hw= (np.matmul(rot, base_move.T))
#     return np.array([-move_hw[1], move_hw[0], degree], dtype='f')

def move(currState, r, L, UL, UR, obsChk, fact, explore=0, chk = True, plot= False, clr='blue'):
    t = 0
    dt = 0.1
    Xn= currState[1]
    Yn= 10.0-currState[0]   # origin shift to bottom left from top left
    Thetan = 3.14 * currState[2] / 180
    status = True
    final = None
    dis = 0
    itr =0
    while (itr<10 and chk):
        itr+=1
        t = t + dt
        Xs = Xn
        Ys = Yn
        dx = 0.5*r * (UL + UR) * math.cos(Thetan) * dt
        dy = 0.5*r * (UL + UR) * math.sin(Thetan) * dt
        # dis += np.linalg.norm((dx, dy))
        dis += np.sqrt(dx**2 + dy**2)
        Xn += dx
        Yn += dy
        Thetan += (r / L) * (UR - UL) * dt
        if(not obsChk.notObs(10.0-Yn, Xn)):
            status = False
            dis =-1
            break
        elif(explore):
            plt.plot([Xs*fact, Xn*fact], [(10-Ys)*fact, (10-Yn)*fact], color=clr)

    if(not obsChk.notObs(10.0-Yn, Xn)):
        status = False
    
    if(plot):
        # Thetan = 180 * (Thetan) / 3.14
        # final = np.array([Xn, 10.0-Yn, Thetan])
        t= 0
        itr =0
        while (itr<10):
            itr+=1
            t = t + dt
            Xs = Xn
            Ys = Yn
            dx = 0.5*r * (UL + UR) * math.cos(Thetan) * dt
            dy = 0.5*r * (UL + UR) * math.sin(Thetan) * dt
            # dis += np.linalg.norm((dx, dy))
            Xn += dx
            Yn += dy
            Thetan += (r / L) * (UR - UL) * dt
            plt.plot([Xs*fact, Xn*fact], [(10-Ys)*fact, (10-Yn)*fact], color=clr)
            # plt.pause(0.000001)
            # plt.plot([Xs*fact, Xn*fact], [Ys*fact, Yn*fact], color=clr)
            # plt.show()

    Thetan = 180 * (Thetan) / 3.14
    if(Thetan<0):
        Thetan = 360 + Thetan
    elif(Thetan>360):
        Thetan = Thetan%360
        
    final = np.array([10.0-Yn, Xn, Thetan])
#     print('move state', final)
#     print('dis ', dis)
    
    return status, final, dis


# import numpy as np
# import math
# currState= np.array([3,-4,0], dtype='f')
# chk = True
# t = 0.0
# dt = 0.1
# Xn= currState[1]
# Yn= 10.0-currState[0] 
# Thetan = 3.14 * currState[2] / 180
# status = True
# final = None
# dis = 0
# itr=0
# UL = 10
# UR = 10
# r = 0.076/2
# L = 0.354
# ul = UL
# ur = UR
# xx = (r/2)*(ul+ur)
# zz = (r/L)*(ul-ur)
# while (itr<1 and chk):
#     itr+=1
#     t = t + dt
#     Xs = Xn
#     Ys = Yn
#     dx = 0.5*r * (UL + UR) * math.cos(Thetan) * dt
#     dy = 0.5*r * (UL + UR) * math.sin(Thetan) * dt
#     # dis += np.linalg.norm((dx, dy))
#     dis += np.sqrt(dx**2 + dy**2)
#     Xn += dx
#     Yn += dy
#     Thetan += (r / L) * (UR - UL) * dt


# Thetan = 180 * (Thetan) / 3.14
# if(Thetan<0):
#     Thetan = 360 + Thetan
# elif(Thetan>360):
#     Thetan = Thetan%360

# final = np.array([10.0-Yn, Xn, Thetan])

# ul = UL
# ur = UR
# xx = (r/2)*(ul+ur)
# v_diff = (r/2)*(ul-ur)
# zz = v_diff*2/L



# =====SECTION 4: USER INPUT=====
def astar(exp):
    fact = 100
    hc= 5
    wc = -5
    tot = 0
    rad = 0.177 # 0.354/2
    actionSet= []
    checkInp= True
    while checkInp:
        print("Enter the robot's clearance (in meters, max upto 3 decimal places will be considered): ")
        try:
            clr = float(input())
            checkInp= False
        except:
            print 'Wrong input....Try again'
    # print('clearance (over and above robot radius) set at meters: ', clr)
    clr = float(int(clr*1000)/1000.)
    tot= rad+clr
    print('Total clearance from obstacles is set at (meters): ', tot)
    map1 = FinalMap(10, 10, tot, fact)
    map1.circ(1, 5, 5)
    map1.circ(1, 2, 7)
    map1.circ(1, 8, 3)
    map1.circ(1, 8, 7)

    map1.sqr(1.5, (1.25 + 1.5/2), (2.25+ 1.5/2))
    map1.sqr(1.5, 5, 1)
    map1.sqr(1.5, 5, 9)
    map1.sqr(1.5, (1.25 + 1.5/2), (2.25+ 1.5/2))

    obsChk= Obs(10, 10, tot, fact)

    # print('Enter the initial starting coordinates in x as (-5,5)m & y as (-5,5)m; and theta in [0,360) with origin at center')
    # print('Enter in the order of x,y,t [separated by commas]: ')       
    # if(len(cart)!=3):
    #     print('Wrong input....\nNote: Initial has  within the map boundaries (and not in any obstacle)')
    #     print('\nTerminating')
    #     terminate = 0
    #     return
    # else:
    # init= np.array([hc-cart[1], cart[0]-wc, cart[2]], dtype='f')
    # # init_n = threshold_state(init)
    # if(not obsChk.notObs(init[0], init[1])):
    #     print('Start position cannot be in the obstacle.....\nTerminating')
    #     return actionSet


    # Set as point robot

    #correction factor
    


    print("Note: ")
    print("All user data is interpreted as in meters")
    print("Least count of 1cm is considered for the complete system")
    print("with a threshold of 0.5cm in x and y coordinates and 30 degrees in rotation for node generation")
    print("Eg 1.014 m will be treated as 1.010 m or 101.0 cm")
    print("Eg 1.015 m will be treated as 1.015 m or 101.0 cm")
    print("Eg 1.015 m will be treated as 1.016 m or 102.0 cm")

    checkInp= True
    while checkInp:
        print('Enter the initial starting coordinates in x as (-5,5)m & y as (-5,5)m; and theta in [0,360) with origin at center')
        print('Enter in the order of x,y,t [separated by commas]: ')
        cart= np.array(input(), dtype='f')
        if(len(cart)!=3):
            print 'Wrong input....Try again \nNote: Only 3 numbers needed inside the map boundaries'
        else:
            init= np.array([hc-cart[1], cart[0]-wc, cart[2]], dtype='f')
            # init_n = threshold_state(init)
            if(not obsChk.notObs(init[0], init[1])):
                print 'Start position cannot be in the obstacle.....Try again'
            else:
                checkInp = False


    # --- Use Final Map ---
    height = int(10*fact/thres_y)
    width = int(10*fact/thres_x)
    depth = int(360/thres_t)


    # Save map as an jpg image
    cv2.imwrite('grid_init.jpg',map1.grid)
    map2= cv2.imread('grid_init.jpg')
    map2 = cv2.cvtColor(map2, cv2.COLOR_BGR2RGB)
    plt.grid()
    plt.ion()
    plt.imshow(map2)
    plt.show()

    # Get starting position from user


    # checkInp= True
    # while checkInp:
    #     print('Enter the initial starting coordinates in x as (-5,5)m & y as (-5,5)m; and theta in [0,360) with origin at center')
    #     print('Enter in the order of x,y,t [separated by commas]: ')
    #     cart= np.array(input(), dtype='f')
    #     if(len(cart)!=3):
    #         print 'Wrong input....Try again \nNote: Only 3 numbers needed inside the map boundaries'
    #     else:
    #         init= np.array([hc-cart[1], cart[0]-wc, cart[2]], dtype='f')
    #         # init_n = threshold_state(init)
    #         if(not obsChk.notObs(init[0], init[1])):
    #             print 'Start position cannot be in the obstacle.....Try again'
    #         else:
    #             checkInp = False

    # Get step size from user
    checkInp= True
    while checkInp:
        print('Enter the left and right wheel RPM: ')
        rpm= np.array(input(), dtype='f')
        if(len(rpm)!=2):
            print('Wrong input, try again.....')
        else:
            checkInp= False
    rpm = rpm*math.pi*2/60
    # Get goal position from user
    checkInp= True
    while checkInp:
        print('Enter the goal coordinates with origin at the center as x,y [separated by commas]: ')
        fs= np.array(input(), dtype='f')
        if(len(fs)!=2):
            print 'Wrong input....Try again \nNote: Only 2 numbers needed inside the map boundaries'
        else:
            finalState= np.array([hc-fs[1], fs[0]-wc, 0], dtype='f')
            # fs_n = threshold_state([finalState[0], finalState[1], 0], fact)
            if(not obsChk.notObs(finalState[0], finalState[1])):
                print 'Goal position cannot be in the obstacle.....Try again'
            else:
                checkInp = False

    plt.ioff()
    graph= AllNodes(height, width, depth) # graph object created for AllNodes class
    # graph.setCost2go(threshold_state([finalState[0]*fact, finalState[1]*fact, 0]))

    # graph.setCost2go(scaled(finalState,fact))
    parentState= init
    # parentStateScaled= scaled(parentState, fact)
    parentNode= scaled(parentState, fact)
    parentCost= 0
    parent_ownId= 0
    # appending the first node into allNodes data set
    # parentNode= threshold_state(parentStateScaled)

    # graph.visit(parentNode, parentState)
    # graph.updateVisited(parentNode)
    graph.updateCost2Come(parentNode, parentCost, parent_ownId,-1, parentState, finalState)
    found= False
    # reached= goalReached(parentStateScaled, scaled(finalState, fact), fact)
    # reached= goalReached(parentState, finalState, fact)
    # if(reached):
    #     found =True
    #     print('Input position is within the goal region')

    # =====SECTION 5: EXPLORATION=====
    wheel_rad = 0.076/2
    wheel_dis = 0.354

    print('Processing...Please wait')
    plt.ion()
    start_time = time.time()
    # graph.setTotCost()

    actions = [[0, rpm[0]], [rpm[0],0], [rpm[0],rpm[0]], [0,rpm[1]], [rpm[1],0], [rpm[1],rpm[1]], [rpm[0],rpm[1]], [rpm[1],rpm[0]]]

    # graph.totCost[parentNode[0], parentNode[1], parentNode[2]]= graph.totCost[parentNode[0], parentNode[1], parentNode[2]]*(-1)
    itr =0
    flag =0
    while(found != True):
        # current node is termed as parent node 
        itr+=1
        if(exp and itr%20==0):
            plt.pause(0.0000001)
        # data = [self.totCost[cord[0], cord[1], cord[2]], curr[0], curr[1], curr[2], pid]
        
        if(len(graph.sort)==0):
            print('No solution exist, terminating....')
            flag=1
            break

        data = heappop(graph.sort)
        newMin = data[0]
        #         print('in try')
        if(newMin == float('inf')):
            print('No solution exist, terminating....')
            flag=1
            break
        else:
            parentState = np.array([data[1], data[2], data[3]])
            ppid = data[4]
            parentCost = data[5]

        # status, minTotCost, new_parentState, ppid = graph.minCostIdx(actions, fact, obsChk, wheel_rad, wheel_dis) # minCostIdx(self, act, fact, r, L):
    #     print('min tot cost', minTotCost)

        # parentState = new_parentState
        # print('new state', parentState)

        # parentStateScaled = scaled(parentState, fact)
        parentNode = scaled(parentState, fact)

        # parentNode = threshold_state(parentStateScaled)
        # print('new node', parentNode)

        # parentCost = graph.cost2come[parentNode[0], parentNode[1]]

        # graph.updateCost2Come(parentNode, parentCost, parentId)
        # graph.totCost[parentNode[0], parentNode[1], parentNode[2]]= graph.totCost[parentNode[0], parentNode[1], parentNode[2]]*(-1)
        
        parent_ownId = graph.visit(parentNode, parentState)
        # parentId = graph.getOwnId(parentNode)
    #     input()
        reached= goalReached(parentState, finalState, fact)
        if(reached):
            found =True
            print('Solved')
            break


        # print('p id=', parentId)
        # if(itr%100==0):
        #     print('iter= ', itr)
        # plt.pause(0.000001)
    #     print('parent state: ', parentState)
    #     print('parent c2c: ', graph.cost2come[parentNode[0], parentNode[1], parentNode[2]])
    #     print('tot cost', graph.totCost[parentNode[0], parentNode[1], parentNode[2]])

        # print('current node id: ', parent_ownId)
        # print('parent cost2come= ', parentCost)
        # plt.pause(0.000001)
        count =0 
        for action in actions: # Iterating for all possible angles
            chk, tempState, dis = move(parentState, wheel_rad, wheel_dis, action[0], action[1], obsChk, fact, exp)  # move(currState, r, L, UL, UR, obsChk, fact, chk = True, explore=0, plot= False, clr='blue'):
            # if(chk and tempNode[0]>=0 and tempNode[0]<=graph.visited.shape[0] and tempNode[1]>=0 and tempNode[1]<=graph.visited.shape[1] ):
            if(chk):
                tempNode = scaled(tempState, fact)
                # tempNode = threshold_state(tempStateScaled)
                if(graph.visited[tempNode[0], tempNode[1], tempNode[2]]==0):
                    tempCost2Come = parentCost + dis
                   
    #                 print('chk1 ', count)
                    # print('temp node', tempNode)
    #                 print('tcost', tempCost)
                    # graph.actDone[tempNode[0], tempNode[1]] = count
                    graph.updateCost2Come(tempNode, tempCost2Come, parent_ownId, count, tempState, finalState)
                    # input()
                    # if(tempNode[0] == 819 and tempNode[1]==519 and tempNode[2]== 10):
                    #     input('found   array([819, 519,  10]')
    #                 print('new tot', graph.totCost[tempNode[0], tempNode[1], tempNode[2]])
    #                 print('chk1 ', count)
            count += 1
    #         print(count)

            # for sd in range(1,step_d+1):
               #  step = action(sd, angle+parentState[2])
               #  tempState = parentState + step
               #  tempState[2] = tempState[2] - parentState[2]
               #  if(not obsChk.notObs(tempState[0], tempState[1])):
               #    chk = False
               #    break;
            # tempNode = threshold_state(tempState)
            # if(chk and tempNode[0]>=0 and tempNode[0]<=graph.visited.shape[0] and tempNode[1]>=0 and tempNode[1]<=graph.visited.shape[1] ):
            #     if(graph.visited[tempNode[0], tempNode[1], tempNode[2]]==0):
            #         tempCost = parentCost + step_d
            #         graph.updateCost(tempNode, tempCost, parentId)
    #     print(status)
        
        # plotStates(org_parentState, new_parentState)
        # print('min cost parent id', ppid)
        

    # =====SECTION 6: PATH GENERATION=====
    # Print final time
    print("Time explored = %2.3f seconds " % (time.time() - start_time))
    plt.show()
    plt.savefig('fig3.png', bbox_inches='tight')
    plt.ioff()
    # back-tracking for the shortest path
    reached_state = graph.getStates(int(len(graph.allStates))-1)
    reached_stateScaled = scaled(reached_state, fact)
    reached_node = threshold_state(reached_stateScaled)
    ans= graph.getOwnId(reached_node)

    finalPath =[]
    
    while(ans!=0 and flag==0):
        goalState= graph.getStates(ans)
        goalNode = scaled(goalState, fact)
        g_actId = graph.actDone[goalNode[0], goalNode[1], goalNode[2]]

        ans= graph.getParentId(goalNode)
        prevState= graph.getStates(ans)
        prevNode = scaled(prevState, fact)
        p_actId = graph.actDone[prevNode[0], prevNode[1], prevNode[2]]
        finalPath.append([goalState, g_actId])
        actionSet.append(actions[g_actId])

    if(flag==0):
        finalPath.append([prevState, p_actId])
        actionSet.append(actions[p_actId])
        # print '\nYellow area shows all the obstacles and White area is the free space'
        # print 'Black lines show all the explored Nodes (area)'
        # print 'Red line shows optimal path (traced from goal node to start node)'
    plt.ion()
    i= len(finalPath)-1
    while(i >0 and flag==0):
        startState = finalPath[i][0]
        actId =  finalPath[i-1][1]
        step = actions[actId]
        _,_,_ =move(startState, wheel_rad, wheel_dis, step[0], step[1], obsChk, fact, 0, False, True, 'red')
        plt.pause(0.000001)
        i-=1
    plt.savefig('back_tracking.png', bbox_inches='tight')
    print('Enter any NUMBER to exit: ')
    input()
    plt.ioff()
    if(flag==0):
        actionSet.reverse()
    return actionSet

Parser = argparse.ArgumentParser()
Parser.add_argument('--exp', default=0, type =int, help='Set to 1 to plot all explored nodes (default: 0)')
Args = Parser.parse_args()

exp = Args.exp
astar(exp)
