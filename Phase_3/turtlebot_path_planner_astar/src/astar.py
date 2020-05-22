# =====SECTION 1: LIBRARIES=====
import numpy as np
import copy as cp
import math
import matplotlib.pyplot as plt
try:
    import cv2
except:
    import sys
    sys.path.remove(sys.path[1])
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
        """Fixed square shape for obstacle"""
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
        wd:     column dimension [pixels]
        c:      clearance from map border"""
        self.ht = height * fact
        self.wd = width * fact
        clr = clr*fact
        clr = int(math.ceil(clr))
        self.c = clr
        self.f = fact

    def bound(self, x,y):
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
        self.actDone= np.ones([self.h_, self.w_, self.d_], dtype='int8')*(-1)
        self.sort=[]

    # Function to get update cost in cost2come array
    def updateCost2Come(self, cord, cost, pid, actId, currState, goalState):
        # print('array cost of tempNode', self.cost2come[cord[0], cord[1], cord[2]])
        if(self.cost2come[cord[0], cord[1], cord[2]] > cost):
            self.cost2come[cord[0], cord[1], cord[2]] = cost
            c2g = heu(currState,goalState)
            totCost = cost + c2g
            self.pIDarr[cord[0], cord[1], cord[2]] = pid
            self.actDone[cord[0], cord[1], cord[2]] = actId
            data = [totCost, currState[0], currState[1], currState[2], pid, cost]
            heappush(self.sort, data)
        return

    # Function to add new unique node in the Nodes data set
    def visit(self, node, state):
        ownId = int(len(self.allStates))
        self.ownIDarr[node[0], node[1], node[2]] = ownId
        self.visited[node[0], node[1], node[2]] = 1
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

def heu(current, goal): #defining heuristic function as euclidian distance between current node and goal
    h = math.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)
    return h

# Function to check if the goal node is reached
def goalReached(curr, goal, fact):
    rad = 0.05
    dis = np.sqrt((curr[0] - goal[0])**2 + (curr[1] - goal[1])**2)
    if (dis<= rad):
        # print('goal clearance radius ', rad)
        # print('actual accepted distance from goal', dis)
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
        t= 0
        itr =0
        while (itr<10):
            itr+=1
            t = t + dt
            Xs = Xn
            Ys = Yn
            dx = 0.5*r * (UL + UR) * math.cos(Thetan) * dt
            dy = 0.5*r * (UL + UR) * math.sin(Thetan) * dt
            Xn += dx
            Yn += dy
            Thetan += (r / L) * (UR - UL) * dt
            plt.plot([Xs*fact, Xn*fact], [(10-Ys)*fact, (10-Yn)*fact], color=clr)

    Thetan = 180 * (Thetan) / 3.14
    if(Thetan<0):
        Thetan = 360 + Thetan
    elif(Thetan>360):
        Thetan = Thetan%360
        
    final = np.array([10.0-Yn, Xn, Thetan])
    
    return status, final, dis

# =====SECTION 4: USER INPUT=====
def astar(initCord, clrn, exp=0):
    fact = 100
    hc= 5
    wc = -5
    tot = 0
    rad = 0.177 # 0.354/2
    cart= np.array(initCord, dtype='f') 
    actionSet= []
    clrr = np.array(clrn, dtype='f')
    if(clrr[0] >= 0):
        clr = clrr[0]
    else:
        print('Clearnace cannot be negative.....\nTerminating')
        return actionSet
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

    init= np.array([hc-cart[1], cart[0]-wc, cart[2]], dtype='f')
    if(not obsChk.notObs(init[0], init[1])):
        print('Start position cannot be in the obstacle.....\nTerminating')
        return actionSet

    print("Note: ")
    print("All user data is interpreted as in meters")
    print("Least count of 1cm is considered for the complete system")
    print("with a threshold of 1cm in x and y coordinates and 30 degrees in rotation for node generation")
    print("Eg 1.014 m will be treated as 1.010 m or 101.0 cm")
    print("Eg 1.015 m will be treated as 1.015 m or 101.0 cm")
    print("Eg 1.015 m will be treated as 1.016 m or 102.0 cm")

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

    checkInp= True
    while checkInp:
        print('Enter the left and right wheel RPM: ')
        rpm= np.array(input(), dtype='f')
        if(len(rpm)!=2):
            print('Wrong input, try again.....')
        else:
            checkInp= False
    rpm = rpm*math.pi*2/60

    checkInp= True
    while checkInp:
        print('Enter the goal coordinates with origin at the center as x,y [separated by commas]: ')
        fs= np.array(input(), dtype='f')
        if(len(fs)!=2):
            print 'Wrong input....Try again \nNote: Only 2 numbers needed inside the map boundaries'
        else:
            finalState= np.array([hc-fs[1], fs[0]-wc, 0], dtype='f')
            if(not obsChk.notObs(finalState[0], finalState[1])):
                print 'Goal position cannot be in the obstacle.....Try again'
            else:
                checkInp = False

    plt.ioff()
    graph= AllNodes(height, width, depth) # graph object created for AllNodes class
    parentState= init
    parentNode= scaled(parentState, fact)
    parentCost= 0
    parent_ownId= 0
    graph.updateCost2Come(parentNode, parentCost, parent_ownId,-1, parentState, finalState)
    found= False

    # =====SECTION 5: EXPLORATION=====
    wheel_rad = 0.038 #0.076/2
    wheel_dis = 0.354

    print('Processing...Please wait')
    plt.ion()
    start_time = time.time()
    actions = [[0, rpm[0]], [rpm[0],0], [rpm[0],rpm[0]], [0,rpm[1]], [rpm[1],0], [rpm[1],rpm[1]], [rpm[0],rpm[1]], [rpm[1],rpm[0]]]
    itr =0
    flag =0
    while(found != True):
        itr+=1
        if(exp and itr%10==0):
            plt.pause(0.000001)
        
        if(len(graph.sort)==0):
            print('No solution exist, terminating....')
            flag=1
            break

        data = heappop(graph.sort)
        newMin = data[0]
        if(newMin == float('inf')):
            print('No solution exist, terminating....')
            flag=1
            break
        else:
            parentState = np.array([data[1], data[2], data[3]])
            ppid = data[4]
            parentCost = data[5]

        parentNode = scaled(parentState, fact)
        parent_ownId = graph.visit(parentNode, parentState)
        reached= goalReached(parentState, finalState, fact)
        if(reached):
            found =True
            print('Solved')
            break
        count =0 
        for action in actions: # Iterating for all possible actions
            chk, tempState, dis = move(parentState, wheel_rad, wheel_dis, action[0], action[1], obsChk, fact, exp)
            if(chk):
                tempNode = scaled(tempState, fact)
                if(graph.visited[tempNode[0], tempNode[1], tempNode[2]]==0):
                    tempCost2Come = parentCost + dis
                    graph.updateCost2Come(tempNode, tempCost2Come, parent_ownId, count, tempState, finalState)
            count += 1
    print("Time explored = %2.3f seconds " % (time.time() - start_time))
    plt.show()
    plt.ioff()
    reached_state = graph.getStates(int(len(graph.allStates))-1)
    reached_node = scaled(reached_state, fact)
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
    print('Enter any NUMBER to start the simulation in Gazebo: ')
    input()
    plt.ioff()
    if(flag==0):
        actionSet.reverse()
    return actionSet
