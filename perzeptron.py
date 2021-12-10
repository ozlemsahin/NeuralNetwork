#Perzeptron Lernalgorithmus
from random import random
from numpy import dot
import numpy as np
from tabulate import tabulate

#trainingsdata 
#            x0,  x1, x2 
training = [[1.0,1.0,1.0],
            [1.0,1.0,0.0],
            [1.0,0.0,1.0],
            [1.0,0.0,0.0]]

#          y1, y2, y3, y4 
target = [1.0,1.0,1.0,0.0]

weights  = [random() for i in range (0,3)]
learning_rate = [0.5]

#------------------------------------loop starts--------------------------
isThereFalseX = True 
#loop continues till every x is predicted true by weights
while(isThereFalseX):
    for i in range(0,4):
        
        expected = target[i]
        x = training[i] #take each trainingdata per calculation
        activation = dot(x, weights)
        prediction = 1.0 if activation >= 0.0 else 0.0
        
       
       
        isThereFalseX = False
        if prediction != expected:
            #change weight 
            x = np.array(x)
            weights = weights +  int((expected - prediction)) * x * learning_rate
            isThereFalseX = True
            
       
        print(tabulate([["w0", weights[0]],["w1", weights[1]], ["w2", weights[2]], ["x0", x[0]], ["x1", x[1]],["x2", x[2]], ["expected", expected],["prediction",prediction]]))     
       

