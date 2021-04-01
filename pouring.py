import os
import tensorflow as tf
import numpy as np
import evaluation_util
import preprocessor
import model_simple_vel
import matplotlib.pyplot as pt
import matplotlib.animation as animation
import util
import matplotlib.ticker as tk
import scipy.signal as sg
import time

import sim
import sys
import math


def create_model(dim_input, dim_output, unit, layer, seq_len, dropout=0):
    '''
    Params:
    dim_input: dimension of the input
    dim_output: dimension of the output
    unit: number of rnn units
    layer: number of rnn layers
    seq_len: maximum sequence length
    dropout: % of dropout [0,1)

    Returns:
    model: Neural network model
    '''
    model = tf.keras.Sequential()

    # Adds the input dimension and does not take into consideration zero values of the input sequences
    model.add(tf.keras.layers.Input(shape=(seq_len, dim_input)))
    model.add(tf.keras.layers.Masking(mask_value=0))

    # Adds the different rnn layers
    for i in range(layer):
        lstm_cell = tf.keras.experimental.PeepholeLSTMCell(unit)
        model.add(tf.keras.layers.RNN(lstm_cell, return_sequences=True, name='LSTM{}'.format(i)))
        model.add(tf.keras.layers.Dropout(dropout))

    # Adds the output dense layer
    model.add(tf.keras.layers.Dense(dim_output, activation='linear'))

    # Defines the loss and optimizer with default values
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


wd = os.getcwd()
model_dir_path_vel = os.path.join(wd,
                                  'train_simple_vel_partial_e2000lr0.001seed1599086380unit16layer1prepend60dropout0.5regu0.0')

model_filename_vel = evaluation_util.find_lowest_cost_model(model_dir_path_vel, is_train=False)
filepath_vel = os.path.join(model_dir_path_vel, model_filename_vel)

train_ratio = 0.7
valid_ratio = 0.2
prepend_length = 60
seed = 1599086380
preprocessed_data = preprocessor.SimpleVelPreprocessor('data_pour_partial.npy', seed, train_ratio, valid_ratio,
                                                       prepend_len=prepend_length)

# 3d is 6 parameters
print(preprocessed_data.input_train.shape)
print(preprocessed_data.input_valid.shape)

print(preprocessed_data.input_test.shape)
# 3d is target v
print(preprocessed_data.target_train.shape)
print(preprocessed_data.target_valid.shape)
# lengths_test is the valid length
print(preprocessed_data.lengths_test.shape)

dim_input = 6
dim_output = 1
unit = 16
layer = 1
seq_len = preprocessed_data.input_train.shape[1]
lengths = preprocessed_data.lengths_test

# model = create_model(dim_input, dim_output, unit, layer, seq_len)
# # Load the previously saved weights
# model.load_weights(filepath_vel).expect_partial()
#
# model.summary()


seq_len = 1
i = 0
# change batch size
nn_input = tf.keras.layers.Input(shape=(seq_len, dim_input), batch_size=1)
nn_mask = tf.keras.layers.Masking(mask_value=0)(nn_input)
lstm_cell = tf.keras.experimental.PeepholeLSTMCell(unit)
nn_cell, h, c = tf.keras.layers.RNN(lstm_cell, return_sequences=True, return_state=True, stateful=True,
                                    name='LSTM{}'.format(i))(nn_mask)
state = [h, c]
nn_output = tf.keras.layers.Dense(dim_output)(nn_cell)

model = tf.keras.Model(nn_input, [nn_output, state])

model.summary()

# Load the previously saved weights
model.load_weights(filepath_vel).expect_partial()

result = np.zeros((preprocessed_data.input_test.shape[1], preprocessed_data.input_test.shape[0]))
state = np.zeros((preprocessed_data.input_test.shape[1], 2, preprocessed_data.input_test.shape[0], unit))

'''
coppelia part
'''

sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to CoppeliaSim
if clientID != -1:
    print('Connected to remote API server')
else:
    print("fail")
    sys.exit()
returnCode = sim.simxStartSimulation(clientID,sim.simx_opmode_oneshot)
if returnCode!=0 and returnCode!=1:
    print("something is wrong")
    print(returnCode)
    exit(0)
# init the force sensor reading
res, f = sim.simxGetObjectHandle(clientID, 'Force_sensor', sim.simx_opmode_blocking)   # sensor under the cup
returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(clientID, f, sim.simx_opmode_streaming)


res, box = sim.simxGetObjectHandle(clientID, 'box', sim.simx_opmode_blocking)    # senor under the box
returnCode1, state1, forceVector1, torqueVector1 = sim.simxReadForceSensor(clientID, box, sim.simx_opmode_streaming)

# get the handle for the source container
res, pour = sim.simxGetObjectHandle(clientID, 'joint', sim.simx_opmode_blocking)

# height and diameter are in mm
H = 186.45
D = 133.33
# H=123.34
# D=8.
k = 2 / D
friction=0.02
length=0.025
# weight and force reading are originally in N, converted to lbf
single_block_weight = 14.375e-03 * 9.8 / 4.448
number_of_blocks =45
target=number_of_blocks - 40
total_weight = single_block_weight * number_of_blocks
target_weight = single_block_weight * target
# record the weight of the receiving cup, positive number makes more sense
returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(clientID, f, sim.simx_opmode_buffer)
receiver_self_weight = -1 * forceVector[2] / 4.448
#  use the box reading
# this box_self_weight includes the receiver cup's weight
# 31.8198 is the weight for the box and empty receiving cup
returnCode2, state2, forceVector2, torqueVector2 = sim.simxReadForceSensor(clientID, box, sim.simx_opmode_buffer)
box_self_weight = -1 * forceVector2[2] / 4.448


# get the starting position of source 
returnCode, original_position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_streaming)
returnCode, original_position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_buffer)
returnCode, position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_buffer)
# standardization parameters
mean_train_new, std_train_new = preprocessed_data.mean_train, preprocessed_data.std_train
# mean_train_new, std_train_new = util.get_mean_std(preprocessed_data.input_train)
mean = np.empty([1, 1, 6], dtype=float)
mean = mean_train_new
std = np.empty([1, 1, 6], dtype=float)
std = std_train_new
print("mean", mean)
print("std", std)
# construct the 6 parameters array, in the order of H, K, f_total, f_2_pour, force reading, angle displacement
reading = np.empty([1, 1, 6], dtype=float)


# convert radians to degree N-m to lbf
reading[:, :, 0] = original_position * 180 / math.pi
reading[:, :, 1] = 0
reading[:, :, 2] = total_weight
reading[:, :, 3] = target_weight
reading[:, :, 4] = H
reading[:, :, 5] = k

i = 0
force = []
time_step = []


'''
force direction 
f_to_pour and f_total are positive. Force sensor reading is negative. 
'''

patient = 100
data=[]
final_cubes=0


time_list=[]
# loop until read the desire target value
while True:
    # 60HZ
    start = time.perf_counter()

    # standardize the input
    temp = np.subtract(reading, mean)
    inputs = np.divide(temp, std)

    # predict the  angular velocity
    result1, state1 = model(inputs[:, :, :])
    speed = np.squeeze(result1.numpy())
    if abs(position+0.87)<0.002:
        velocity=speed

    #  use the box reading
    returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(clientID, box, sim.simx_opmode_buffer)
    # returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(clientID, f, sim.simx_opmode_buffer)

    # update the reading, only need to change force reading and angle displacement
    # we want to keep the force reading negative, so + the cup weight
    # weight_in_receiver = ((forceVector[2] / 4.448) + receiver_self_weight)

    weight_in_receiver = ((forceVector[2] / 4.448) + box_self_weight)
    print(position)
    if target / number_of_blocks < 0.13:
        if -1 * weight_in_receiver < target_weight and -1.42 < position < -0.9:
            speed=velocity
            print(speed)

    # get force senor reading and joint position
    returnCode, position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_buffer)

    if i < 60:
        i += 1
        errorCode = sim.simxSetJointTargetVelocity(clientID, pour, -0.05, sim.simx_opmode_oneshot)
        continue

    # if already reach the target but the model fail to come back, break
    if abs(-1 * weight_in_receiver - target_weight) < 0.01 or -1 * weight_in_receiver > target_weight:
        if speed<0:
            speed *= -1

    # set the speed
    if position * 180 / math.pi > 45:
        errorCode = sim.simxSetJointTargetVelocity(clientID, pour, 0, sim.simx_opmode_oneshot)
        print(1)
    else:
        errorCode = sim.simxSetJointTargetVelocity(clientID, pour, speed, sim.simx_opmode_oneshot)

    # prevent overshot when rotate back
    if -1*weight_in_receiver > single_block_weight and position>0:
        print("no overshot")
        errorCode = sim.simxSetJointTargetVelocity(clientID, pour, 0, sim.simx_opmode_oneshot)
        final_cubes = round(-1 * weight_in_receiver / single_block_weight)
        print(final_cubes, "Number of Cubes in receiver:box")
        break

    reading[0, 0, 1] = weight_in_receiver
    reading[0, 0, 0] = position * 180 / math.pi

    # collect data
    temp=np.squeeze(inputs)
    temp=np.append(temp,speed)
    data.append(temp)

    force.append(weight_in_receiver)
    time_step.append(i)
    i += 1
    p = position

    # frequency adjustor
    duration = time.perf_counter() - start
    time_list.append(duration)
    cap = 0.01667
    # cap = 0.01

    if duration < cap:
        time.sleep(cap - duration)  # (= 100 Hz)


# if the model doesn't put the cup back to original position after exiting the loop
returnCode, position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_buffer)
if position <0:
    returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(clientID, box, sim.simx_opmode_buffer)
    weight_in_receiver = ((forceVector[2] / 4.448) + box_self_weight)
    final_cubes = round(-1 * weight_in_receiver / single_block_weight)
    print("manually rotate back")
    print(final_cubes, "Number of Cubes in receiver:box")
    while position<0:
        errorCode = sim.simxSetJointTargetVelocity(clientID, pour, 0.2, sim.simx_opmode_oneshot)
        returnCode, position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_buffer)

sim.simxSetJointTargetVelocity(clientID, pour, 0, sim.simx_opmode_oneshot_wait)


# from plot import plot_ori
# plot_ori(data)
#
# pt.title('cap-duration')
# pt.plot(sub,'.')
# pt.show()
#
# pt.title('before wait')
# pt.plot(time_list,'.')
# pt.show()
#
# pt.title('one iteration elapse')
# pt.plot(total_time,'.')
# pt.show()

# calculate how many cubes spilled out
returnCode1, state1, forceVector1, torqueVector1 = sim.simxReadForceSensor(clientID, box, sim.simx_opmode_buffer)
returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(clientID, f, sim.simx_opmode_buffer)

# cubes in the cup
weight_in_receiver = ((forceVector[2] / 4.448) + receiver_self_weight)
cubes_in_cup=round(-1 * weight_in_receiver / single_block_weight)

# final_cubes is the count of cubes in the box
include_spill = final_cubes-cubes_in_cup
print('{} cubes spilled'.format(include_spill))
path='./testing/'
file_name="number_of_blocks"+str(number_of_blocks)+"_target"+str(target)+"_result"+str(final_cubes)+"_friction"+str(friction)+"_length"+str(length)+"_spill"+str(include_spill)
testing=os.path.join(path,file_name+".npy")
with open(testing,'wb') as f:

        np.save(f,np.array(data))
errorCode = sim.simxSetJointTargetVelocity(clientID, pour, 0, sim.simx_opmode_oneshot_wait)
x = sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)

# print("all done, spilled", include_spill)
print("all done, stop sim")
# close the connection to CoppeliaSim


sim.simxFinish(clientID)


'''
# turn the source cup back
while abs(original_position-position)>0.02:    
     
    errorCode=sim.simxSetJointTargetVelocity(clientID,pour,0.2, sim.simx_opmode_oneshot)
    returnCode,position=sim.simxGetJointPosition(clientID,pour,sim.simx_opmode_buffer)
     
    returnCode, state, forceVector, torqueVector=sim.simxReadForceSensor(clientID, f,sim.simx_opmode_buffer)
    weight_in_receiver=((forceVector[2]/4.448)+receiver_self_weight)
    force.append(weight_in_receiver)
    time_step.append(i)
    i+=1   
    result.append(0.2)
    time.sleep(0.016) # (= 60 Hz)
    
pt.plot(time_step,force)
pt.show()	
'''
