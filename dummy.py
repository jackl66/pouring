# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 20:36:03 2020

@author: jackl
"""


import sim
import sys
import time
 
import matplotlib.pyplot as plt   #used for image plotting

sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
if clientID!=-1:
    print ('Connected to remote API server')
else:
    print("fail")
    sys.exit();
    
# testing    
res,objs=sim.simxGetObjectHandle(clientID,'Revolute_joint',sim.simx_opmode_blocking)
sim.simxSetJointTargetVelocity(clientID,objs,0.2,sim. simx_opmode_streaming)


# init the force sensor reading
res,f=sim.simxGetObjectHandle(clientID,'Force_sensor',sim.simx_opmode_blocking)
returnCode, state, forceVector, torqueVector=sim.simxReadForceSensor(clientID, f,sim.simx_opmode_streaming)

# get the handle for the source
res,pour=sim.simxGetObjectHandle(clientID,'joint',sim.simx_opmode_blocking)



# get geometry INFO
# unit is meter
H=1.2334e-01
D=8.8176e-02
k=D/2
# unit is kg
single_block_weight=2.197e-03
number_of_blocks=5
total_weight=single_block_weight*number_of_blocks
target_weight=single_block_weight*(number_of_blocks-1)
# record the weight of the receving cup
returnCode, state, forceVector, torqueVector=sim.simxReadForceSensor(clientID, f,sim.simx_opmode_buffer)
receiver_self_weight=-1*forceVector[2]


# get the starting position of source 
returnCode,original_position=sim.simxGetJointPosition(clientID,pour,sim.simx_opmode_streaming )
returnCode,original_position=sim.simxGetJointPosition(clientID,pour,sim.simx_opmode_buffer)

i=0
force=[]
time_step=[]
# loop until read the desire target value
while -1*forceVector[2]-receiver_self_weight<target_weight:
    errorCode=sim.simxSetJointTargetVelocity(clientID,pour,0.015, sim.simx_opmode_oneshot)
    returnCode, state, forceVector, torqueVector=sim.simxReadForceSensor(clientID, f,sim.simx_opmode_buffer)
    # convert N-m to lbf
    weight_in_receiver=((-1*forceVector[2])-receiver_self_weight)*0.74
    force.append(weight_in_receiver)
    
    time_step.append(i)
    i+=1
    returnCode,position=sim.simxGetJointPosition(clientID,pour,sim.simx_opmode_buffer)

    time.sleep(0.016) # (= 60 Hz)
    
# converge state
k=10
while k>0:    
    returnCode, state, forceVector, torqueVector=sim.simxReadForceSensor(clientID, f,sim.simx_opmode_buffer)
    weight_in_receiver=((-1*forceVector[2])-receiver_self_weight)*0.74
    force.append(weight_in_receiver)
    time_step.append(i)
    i+=1   
    k-=1
    time.sleep(0.016) # (= 60 Hz)
    
    
plt.plot(time_step,force)
plt.show()   
# close the force sensor connection 
sim.simxReadForceSensor(clientID, f,sim.simx_opmode_discontinue)
  
 
# turn the source cup back
while original_position-position<0.02:    
    errorCode=sim.simxSetJointTargetVelocity(clientID,pour,-0.05, sim.simx_opmode_oneshot)
    returnCode,position=sim.simxGetJointPosition(clientID,pour,sim.simx_opmode_buffer)
    time.sleep(0.016) # (= 60 Hz)

print("out")
errorCode=sim.simxSetJointTargetVelocity(clientID,pour,0, sim.simx_opmode_oneshot)

# Now close the connection to CoppeliaSim:
sim.simxFinish(clientID)
