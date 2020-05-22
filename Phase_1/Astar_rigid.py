# =====SECTION 1: LIBRARIES=====
import numpy as np
import copy as cp
import math
import matplotlib.pyplot as plt
import cv2
import time

# =====SECTION 2: MAPS=====

# Final fixed map that has five obstacles
class FinalMap():
    def __init__(self, height, width, clr):
        """Initializes final map
        height:     row dimension [pixels]
        width:      column dimension [pixels]
        c:        clearance from map border"""
        height+=1
        width+=1
        self.c = clr
        self.grid= np.ones([height,width,3], dtype ='uint8')*255
        self.grid[0:(clr+1),:,0] = 0
        self.grid[height-(clr+1):height,:,0] = 0
        self.grid[:,0:(clr+1),0] = 0
        self.grid[:, width-(clr+1):width,0] = 0

    # Obstacle in top left
    def shape1(self):
        """Fixed polygon shape for obstacle in top-left of map"""
        m1, c1 = 13.0, -140
        m2, c2 = 0, 185
        m3, c3 = 7.0/5, 80
        m4, c4 = 1, 100
        m5, c5 = -(7.0/5), 290
        m6, c6 = (6.0/5), 30
        m7, c7 = -(6.0/5), 210
        for x in range(self.grid.shape[1]/2):
            for y in range (self.grid.shape[0]):
                y1= m1*x + c1 + (self.c)*math.sqrt(1+(m1**2))
                y2= m2*x + c2 + (self.c)*math.sqrt(1+(m2**2))
                y3= m3*x + c3 - (self.c)*math.sqrt(1+(m3**2))
                y4= m4*x + c4 - (self.c)*math.sqrt(1+(m4**2))
                if(y<=y1 and y<=y2 and y>=y3 and y>=y4):
                    self.grid[self.grid.shape[0]-1-y,x, 0]=0
                y3a= m3*x + c3 + (self.c)*math.sqrt(1+(m3**2))
                y5= m5*x + c5 + (self.c)*math.sqrt(1+(m5**2))
                y6= m6*x + c6 - (self.c)*math.sqrt(1+(m6**2))
                y7= m7*x + c7 - (self.c)*math.sqrt(1+(m7**2))
                if(y<=y3a and y<=y5 and y>=y6 and y>=y7):
                    self.grid[self.grid.shape[0]-1-y,x, 0]=0

                if(y<=y2 and y>y5 and y>(self.c) and y<(self.grid.shape[0]-(self.c+1))):
                    self.grid[self.grid.shape[0]-1-y,x, 0]=255

                if(y>y2 and y<y3a and y>(self.c) and y<(self.grid.shape[0]-(self.c+1))):
                    self.grid[self.grid.shape[0]-1-y,x, 0]=255
        # Highlighting the vertices of the polygons
        self.grid[self.grid.shape[0]-1-120,75,0:2]= 0
        self.grid[self.grid.shape[0]-1-185,25,0:2]= 0
        self.grid[self.grid.shape[0]-1-185,75,0:2]= 0
        self.grid[self.grid.shape[0]-1-150,100,0:2]= 0
        self.grid[self.grid.shape[0]-1-150,50,0:2]= 0
        self.grid[self.grid.shape[0]-1-120,20,0:2]= 0
        return

    # Obstacle in center
    def ellipse(self, major, minor, h, w):
        """Customiazble ellipse obstacle
            major:  major axis dimension [pixels]
            minor:  minor axis dimension [pixels]
            h:      ellipse center location in map's coordinate system
            w:      ellipse center location in map's coordinate system"""
        finalMajor= major + self.c
        finalMinor= minor + self.c
        if(h- finalMinor <=0):
            ha= 0
        else:
            ha= h - finalMinor
        if(h+finalMinor >= self.grid.shape[0]):
            hb= self.grid.shape[0]
        else:
            hb= h + finalMinor

        if(w- finalMajor <=0):
            wa= 0
        else:
            wa= w-finalMajor

        if(w+finalMajor >= self.grid.shape[1]):
            wb= self.grid.shape[1]
        else:
            wb= w + finalMajor

        for i in range(wa, wb):
            for j in range(ha, hb):
                if ((float(i - w) / finalMajor) ** 2 + (float(j - h) / finalMinor) ** 2) <= 1:
                    self.grid[200-j, i, 0] = 0
        return

    # Obstacle in top right
    def circ(self, radius, h, w):
        """Customizable circle obstacle
            radius:     radius dimention [pixels]
            h:          circle center location in map's coordinate system
            w:          circle center location in map's coordinate system"""
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

    # Obstacle in bottom right
    def rohmbus(self):
        """Fixed polygon shape for obstacle in bottom-right of map"""
        m1, c1 = -(3.0/5), 295
        m2, c2 = (3.0/5), 25
        m3, c3 = -(3.0/5), 325
        m4, c4 = (3.0/5), 55

        # for y in range(self.grid.shape[0]):
        for x in range(self.grid.shape[1]):
            for y in range (self.grid.shape[0]):
                y1= m1*x + c1 - (self.c)*math.sqrt(1+(m1**2))
                y2= m2*x + c2 - (self.c)*math.sqrt(1+(m2**2))
                y3= m3*x + c3 + (self.c)*math.sqrt(1+(m3**2))
                y4= m4*x + c4 + (self.c)*math.sqrt(1+(m4**2))
                if(y>=y1 and y>=y2 and y<=y3 and y<=y4):
                    self.grid[y,x, 0]=0
        # Highlighting the rohmbus vertices
        self.grid[175,200,0:2]= 0
        self.grid[160,225,0:2]= 0
        self.grid[175,250,0:2]= 0
        self.grid[190,225,0:2]= 0
        return

    # Obstacle in bottom left
    def rect(self):
        """Fixed polygon shape for obstacle in bottom-left of map"""
        m1, c1 = -(9.0/5), 186
        m2, c2 = (38.0/65), (1333.0/13)
        m3, c3 = -(9.0/5), 341
        m4, c4 = (38.0/65), (1488.0/13)
        for x in range(self.grid.shape[1]):
            for y in range (self.grid.shape[0]):
                y1= m1*x + c1 - (self.c)*math.sqrt(1+(m1**2))
                y2= m2*x + c2 - (self.c)*math.sqrt(1+(m2**2))
                y3= m3*x + c3 + (self.c)*math.sqrt(1+(m3**2))
                y4= m4*x + c4 + (self.c)*math.sqrt(1+(m4**2))
                if(y>=y1 and y>=y2 and y<=y3 and y<=y4):
                    self.grid[y,x, 0]=0
        # Highlighting the rectangle vertices
        self.grid[132,30,0:2]= 0
        self.grid[123,35,0:2]= 0
        self.grid[161,100,0:2]= 0
        self.grid[170,95,0:2]= 0
        return

# Class to have map representation as math equations and methods to check if a point
# is in free space or in the obstacle
class Isobs():
    def __init__(self, height, width, clr):
        """Initializes final map
        ht:     row dimension [pixels]
        wd:      column dimension [pixels]
        c:        clearance from map border"""
        self.ht=height+1
        self.wd=width+1
        self.c = clr

    def bound(self, x,y):
    	if((y>=0 and y<=self.c) or (y>=(self.ht-(self.c+1)) and y<self.ht)):
    		return False
    	if((x>=0 and x<=self.c) or (x>=(self.wd-(self.c+1)) and x<self.wd)):
    		return False
    	return True

    def shape1(self, x, y):
        y=200-y
        m1, c1 = 13.0, -140
        m2, c2 = 0, 185
        m3, c3 = 7.0/5, 80
        m4, c4 = 1, 100
        m5, c5 = -(7.0/5), 290
        m6, c6 = (6.0/5), 30
        m7, c7 = -(6.0/5), 210

        cc1, cc1a, cc2, cc2a = False, False, False, False
        y1= (m1*x) + c1 + ((self.c)*math.sqrt(1+(m1**2)))
        y2= (m2*x) + c2 + ((self.c)*math.sqrt(1+(m2**2)))
        y3= (m3*x) + c3 - ((self.c)*math.sqrt(1+(m3**2)))
        y4= (m4*x) + c4 - ((self.c)*math.sqrt(1+(m4**2)))

        if(y<=y1 and y<=y2 and y>=y3 and y>=y4):
            cc1= True
        y3a= m3*x + c3 + (self.c)*math.sqrt(1+(m3**2))
        y5= m5*x + c5 + (self.c)*math.sqrt(1+(m5**2))
        y6= m6*x + c6 - (self.c)*math.sqrt(1+(m6**2))
        y7= m7*x + c7 - (self.c)*math.sqrt(1+(m7**2))
        if(y<=y3a and y<=y5 and y>=y6 and y>=y7):
            cc2= True

        if(y<=y2 and y>y5 and y>(self.c) and y<(self.ht-(self.c+1))):
            cc1a= True

        if(y>y2 and y<y3a and y>(self.c) and y<(self.ht-(self.c+1))):
            cc2a= True
             
        if((cc1 and not cc1a) or (cc2 and not cc2a)):
            return False
        else:
            return True

    def ellipse(self, major, minor, h, w, x, y):
        finalMajor= major + self.c
        finalMinor= minor + self.c
        if ((float(x - w) / finalMajor) ** 2 + (float(y - h) / finalMinor) ** 2) <= 1:
            return False
        return True

    def circ(self, radius, h, w, x, y):
        finalRad = radius + self.c
        eqn= (y-h)**2 + (x-w)**2
        if(eqn<=(finalRad**2)):
            return False
        return True

    def rohmbus(self, x,y):
        m1, c1 = -(3.0/5), 295
        m2, c2 = (3.0/5), 25
        m3, c3 = -(3.0/5), 325
        m4, c4 = (3.0/5), 55
        y1= m1*x + c1 - (self.c)*math.sqrt(1+(m1**2))
        y2= m2*x + c2 - (self.c)*math.sqrt(1+(m2**2))
        y3= m3*x + c3 + (self.c)*math.sqrt(1+(m3**2))
        y4= m4*x + c4 + (self.c)*math.sqrt(1+(m4**2))
        if(y>=y1 and y>=y2 and y<=y3 and y<=y4):
            return False
        return True

    def rect(self, x, y):
        m1, c1 = -(9.0/5), 186
        m2, c2 = (38.0/65), (1333.0/13)
        m3, c3 = -(9.0/5), 341
        m4, c4 = (38.0/65), (1488.0/13)
        y1= m1*x + c1 - (self.c)*math.sqrt(1+(m1**2))
        y2= m2*x + c2 - (self.c)*math.sqrt(1+(m2**2))
        y3= m3*x + c3 + (self.c)*math.sqrt(1+(m3**2))
        y4= m4*x + c4 + (self.c)*math.sqrt(1+(m4**2))
        if(y>=y1 and y>=y2 and y<=y3 and y<=y4):
            return False
        return True

    def notObs(self, y, x):
    	if(self.bound(x,y) and self.shape1(x,y) and self.ellipse(40, 20, 100, 150, x,y) and self.circ(25, 50, 225, x,y) and self.rohmbus(x,y) and self.rect(x,y)):
    		return True
    	else:
    		return False


# =====SECTION 3: ACTIONS AND NODES=====
class AllNodes():
    # Initialize class object
    def __init__(self, height, width, depth):
        """Initializes to keep track of all nodes explored"""
        self.h_ = height + 1
        self.w_ = width + 1
        self.d_ = depth
        self.allStates=[]
        self.visited= np.zeros([self.h_, self.w_, self.d_])
        self.ownIDarr= np.ones([self.h_, self.w_, self.d_], dtype='int64')*(-1)
        self.pIDarr= np.ones([self.h_, self.w_, self.d_], dtype='int64')*(-1)
        self.cost2come= np.ones([self.h_, self.w_, self.d_], dtype='f')*(float('inf'))
        self.cost2go= np.zeros([self.h_, self.w_], dtype='f')
        self.totCost= np.zeros([self.h_, self.w_, self.d_], dtype='f')

    # Function to mark the node as visited in the visited array
    def setCost2go(self, goal):
    	# print('goal', goal)
    	for h in range(self.h_):
    		for w in range(self.w_):
    			dis= math.sqrt((goal[0]-h)**2 + (goal[1]-w)**2)
    			self.cost2go[h,w] = dis
        return

    # Function to initialize total cost as sum of cost2go and cost2come
    def setTotCost(self):
    	for i in range (self.d_):
    		self.totCost[:,:,i] = self.cost2come[:,:,i] + self.cost2go

    # Function to mark the node as visited in the visited array
    def updateVisited(self, cord):
        self.visited[cord[0], cord[1], cord[2]] = 1
        return

    # Function to get update cost in cost2come array
    def updateCost(self, cord, cost, pid):
        if(self.cost2come[cord[0], cord[1], cord[2]] > cost):
            self.cost2come[cord[0], cord[1], cord[2]] = cost
            self.totCost[cord[0], cord[1], cord[2]] = cost + self.cost2go[cord[0], cord[1]]
            self.pIDarr[cord[0], cord[1], cord[2]] = pid
        return

    # Function to add new unique node in the Nodes data set
    def push(self, node, state):
    	self.ownIDarr[node[0], node[1], node[2]] = int(len(self.allStates))
        self.allStates.append(state)
        return

    # Function to get own id
    def getOwnId(self,cord):
        return self.ownIDarr[cord[0], cord[1], cord[2]]

    # Function to get parent id
    def getParentId(self,cord):
        return self.pIDarr[cord[0], cord[1], cord[2]]

    # Function to get state of the node i.e. coordinate [h,w[]
    def getStates(self, idx):
        return self.allStates[idx]

    # Function to get the index value of cost2come array having minimum cost value
    def minCostIdx(self, step):
    	# print 'minimum'	
        try:
            newMin= np.min(self.totCost[self.totCost>0])
            if(newMin == float('inf')):
                status= False
                new_parentState = -1
                p_state = -1
            else:
                status= True
                index= np.argwhere(self.totCost==newMin)[0]
                idx = self.pIDarr[index[0],index[1],index[2]]
                p_state = self.getStates(idx)
                if(p_state[2] - (30*(p_state[2]//30)) >= 15):
                    temp = (30*(p_state[2]//30)) + 30
                else:
                    temp = (30*(p_state[2]//30))

                for a in range (0,360,30):
                    temp_angle = ((temp + a)/30)%12
                    if(temp_angle == index[2]):
                        degree= a;
                        break
                new_parentState = p_state+ action(step, degree+p_state[2])
                new_parentState[2] = new_parentState[2] - p_state[2]
        except:
            newMin = float ('inf')
            status= False
            new_parentState = -1
            p_state = -1
        return status, newMin, new_parentState, p_state

# Function to convert float point states to respective regions in node format
def threshold_state(state):
    node= np.ones(3, dtype='f')*(-1)
    if(state[0] - int(state[0]) < 0.5):
        node[0] = int(state[0])
    elif(state[0] - int(state[0]) >= 0.5 and state[0] - int(state[0]) <= 0.7):
        node[0] = int(state[0]) + 0.5
    else:
        node[0] = int(state[0]) + 1

    if(state[1] - int(state[1]) < 0.5):
        node[1] = int(state[1])
    elif(state[1] - int(state[1]) >= 0.5 and state[1] - int(state[1]) <= 0.7):
        node[1] = int(state[1]) + 0.5
    else:
        node[1] = int(state[1]) + 1

    if(state[2] - (30*(state[2]//30)) >= 15):
        node[2] = (30*(state[2]//30)) + 30
    else:
        node[2] = (30*(state[2]//30))
    node[0]= node[0]*2
    node[1]= node[1]*2
    node[2]= node[2]/30
    return np.array(node, dtype='int32')

# Function to check if the goal node is reached
def goalReached(curr, goal):
    rad = 1.5
    dis = np.sqrt((curr[0] - goal[0])**2 + (curr[1] - goal[1])**2)
    if (dis<= rad):
        return 1
    else:
        return 0

def plotStates(src, dst):
    X0 = src[1]
    Y0 = src[0]
    U0 = dst[1]-X0
    V0 = dst[0]-Y0
    plt.quiver(X0, Y0, U0, -V0, units='xy' ,scale=1,headwidth = 1, headlength=0)
    return

def plotBack(src, dst):
    X0 = src[1]
    Y0 = src[0]
    U0 = dst[1]-X0
    V0 = dst[0]-Y0
    plt.quiver(X0, Y0, U0, -V0, units='xy' , scale=1, color= 'r', headwidth = 1, headlength=0)
    return

# Function to take a step in the direction of speficied angle degree
def action(step, degree):
    if(degree >=360):
        degree = degree-360
    t= math.radians(degree)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype= 'f')
    base_move = np.array([step,0], dtype= 'f')
    move_hw= (np.matmul(rot, base_move.T))
    return np.array([-move_hw[1], move_hw[0], degree], dtype='f')


# =====SECTION 4: USER INPUT=====

# Set as point robot
checkInp= True
while checkInp:
    print('Enter the radius and clearance (enter 0,0 for point robot) [separated by commas]: ')
    print('Ceil value of radius + clearance will be considered')
    a = input()
    if(len(a)==2):
        rad= a[0]
        clr= a[1]
        checkInp= False
    else:
        print 'Wrong input....Try again'
tot= int(math.ceil(rad+clr))

# --- Use Final Map ---
thres_x, thres_y, thres_t = 0.5, 0.5, 30.0
height = int(200/thres_y)
width = int(300/thres_x)
depth = int(360/thres_t)
map1= FinalMap(200, 300, tot)
map1.shape1()
map1.circ(25, 50, 225)
map1.ellipse(40, 20, 100, 150)
map1.rohmbus()
map1.rect()

obsChk= Isobs(200, 300, tot)

# Save map as an jpg image
cv2.imwrite('grid_init.jpg',map1.grid)
map2= cv2.imread('grid_init.jpg')
map2 = cv2.cvtColor(map2, cv2.COLOR_BGR2RGB)
plt.ion()
plt.imshow(map2)
plt.show()
graph= AllNodes(height, width, depth) # graph object created for AllNodes class

# Get starting position from user
checkInp= True
while checkInp:
    print('Enter the initial starting coordinates and theta [0,360) with origin at bottom left as x,y,t [separated by commas]: ')
    cart= np.array(input(), dtype='f')
    if(len(cart)!=3):
        print 'Wrong input....Try again \nNote: Only 3 positive numbers needed inside the map boundaries'
    else:
    	init= np.array([200-cart[1], cart[0], cart[2]], dtype='f')
    	init_n = threshold_state(init)
        if(not obsChk.notObs(init[0], init[1])):
            print 'Start position cannot be in the obstacle.....Try again'
        else:
            checkInp = False

# Get step size from user
checkInp= True
while checkInp:
    print('Enter the step size "d" in integer (1<=d<=10): ')
    step_d= int(input())
    if(step_d<1 or step_d>10):
        print('Wrong step size, try again.....')
    else:
        checkInp= False

# Get goal position from user
checkInp= True
while checkInp:
    print('Enter the goal coordinates with origin at bottom left as x,y [separated by commas]: ')
    fs= np.array(input(), dtype='f')
    if(len(fs)!=2):
        print 'Wrong input....Try again \nNote: Only 2 positive numbers needed inside the map boundaries'
    else:
		finalState= np.array([200-fs[1], fs[0]], dtype='f')
		fs_n = threshold_state([finalState[0], finalState[1],0])
		if(not obsChk.notObs(finalState[0], finalState[1])):
			print 'Goal position cannot be in the obstacle.....Try again'
		else:
			checkInp = False

plt.ioff()
graph.setCost2go(fs_n)
parentState= init
parentCost= 0
parentId= 0
# appending the first node into allNodes data set
parentNode= threshold_state(parentState)
graph.push(parentNode, parentState)
graph.pIDarr[parentNode[0], parentNode[1], parentNode[2]]=0
graph.updateVisited(parentNode)
graph.updateCost(parentNode, parentCost, parentId)
found= False

# check if initial state is same as final state
reached= goalReached(parentState, finalState)
if(reached):
    found =True
    print('Input position is within the goal region')

# =====SECTION 5: EXPLORATION=====
start_time = time.time()
print('Processing...Please wait')
count = 0
plt.ion()
minCost= 0.0
graph.setTotCost()
graph.totCost[parentNode[0], parentNode[1], parentNode[2]]= graph.totCost[parentNode[0], parentNode[1], parentNode[2]]*(-1)
while(found != True):
    # current node is termed as parent node    
    for angle in range(0,360,30): # Iterating for all possible angles
        chk =True
        for sd in range(1,step_d+1):
	        step = action(sd, angle+parentState[2])
	        tempState = parentState + step
	        tempState[2] = tempState[2] - parentState[2]
	        if(not obsChk.notObs(tempState[0], tempState[1])):
	        	chk = False
	        	break;
        tempNode = threshold_state(tempState)
        if(chk and tempNode[0]>=0 and tempNode[0]<=graph.visited.shape[0] and tempNode[1]>=0 and tempNode[1]<=graph.visited.shape[1] ):
            if(graph.visited[tempNode[0], tempNode[1], tempNode[2]]==0):
                tempCost = parentCost + step_d
                graph.updateCost(tempNode, tempCost, parentId)

    status, minCost, new_parentState, org_parentState = graph.minCostIdx(step_d)
    if(not status):
        print('No solution exist, terminating....')
        count=1
        break
    plotStates(org_parentState, new_parentState)
    parentState = new_parentState
    parentNode = threshold_state(parentState)
    parentCost = graph.cost2come[parentNode[0], parentNode[1], parentNode[2]]
    graph.updateCost(parentNode, parentCost, parentId)
    graph.totCost[parentNode[0], parentNode[1], parentNode[2]]= graph.totCost[parentNode[0], parentNode[1], parentNode[2]]*(-1)
    graph.push(parentNode, parentState)
    graph.updateVisited(parentNode)
    parentId = graph.getOwnId(parentNode)
    reached= goalReached(parentState, finalState)
    if(reached):
        found =True
        print('Solved')
        break


# =====SECTION 6: PATH GENERATION=====
# Print final time
print("Time explored = %2.3f seconds " % (time.time() - start_time))
plt.show()
plt.savefig('fig3.png', bbox_inches='tight')

# back-tracking for the shortest path
reached_state = graph.getStates(int(len(graph.allStates))-1)
reached_node = threshold_state(reached_state)
ans= graph.getOwnId(reached_node)
if(not count):
    print '\nYellow area shows all the obstacles and White area is the free space'
    print 'Black lines show all the explored Nodes (area)'
    print 'Red line shows optimal path (traced from goal node to start node)'
while(ans!=0 and count==0):
    startState= graph.getStates(ans)
    startNode = threshold_state(startState)
    ans= graph.getParentId(startNode)
    nextState= graph.getStates(ans)
    nextNode = threshold_state(nextState)
    plotBack(startState, nextState)
    plt.pause(0.000001)

plt.show()
plt.savefig('back_tracking.png', bbox_inches='tight')
print('Enter any NUMBER to exit: ')
input()
plt.ioff()
