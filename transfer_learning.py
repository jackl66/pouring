import os
import tensorflow as tf
import numpy as np
import evaluation_util
import preprocessor
import model_simple_vel
from plot import plotting

import matplotlib.pyplot as pt
import matplotlib.animation as animation
import util
import time
import sim
import sys
import math

mean = [-3.32661193e+01, -7.49158081e-02, 3.53722538e-01, 1.40109484e-01,
        9.39063455e+01, 2.80391889e-02]
std = [2.57054564e+01, 1.03371712e-01, 2.65721940e-01, 1.20684726e-01,
       6.36882466e+01, 4.93812157e-03]
# data=np.load("number_of_blocks50_target2_result2.001974787877121_friction0.06_length0.025.npy")
trails = os.listdir("training")
train_ratio = 0.7
valid_ratio = 0.2
prepend_length = 60
seed = 1599086380

# 82 is the number of trail, 300 is the uniform length for all trails, 6 is the number of features.
final_data_train = np.zeros((82, 300, 6))

# 1 is the actual velocity
final_data_target = np.zeros((82, 300, 1))

'''
load trail one by one, apply the following:
    1. apply mean and std ----Already done during data collection 
    2. adjust all trails to length 300
    3. prepend 60
    4. set initial, target, height,kappa to be the same value 
    2,3 are done by the above initialization. Start filling in the final_data_train at index 60
'''
idx = 0
for trail in trails:
    prepended = 0
    # 4 parameters that don't change over time
    initial = 0
    target = 0
    height = 0
    kappa = 0

    file_path = os.path.join("training", trail)
    data = np.load(file_path)
    print(trail)
    print(data.shape)
    print(data[0, :6])
    # index 20 is randomly picked
    initial = data[20, 2]
    target = data[20, 3]
    height = data[20, 4]
    kappa = data[20, 5]
    print(initial,target,height,kappa)
    for time in range(data.shape[0]):
        final_data_train[idx, time, :] = data[time, :6]
        final_data_target[idx, time, 0] = data[time, -1]

    final_data_train=preprocessor.prepend_data(final_data_train,prepend_length)
    final_data_target = preprocessor.prepend_data(final_data_target, prepend_length)
    # set constants parameters
    final_data_train[idx, :, 2] = initial
    final_data_train[idx, :, 3] = target
    final_data_train[idx, :, 4] = height
    final_data_train[idx, :, 5] = kappa
    idx += 1
    break
print(final_data_target.shape)
print(final_data_train.shape)
plotting(final_data_train[0], final_data_target[0])



# wd = os.getcwd()
# model_dir_path_vel = os.path.join(wd,
#                                   'train_simple_vel_partial_e2000lr0.001seed1599086380unit16layer1prepend60dropout0.5regu0.0')
# model_filename_vel = evaluation_util.find_lowest_cost_model(model_dir_path_vel, is_train=False)
# filepath_vel = os.path.join(model_dir_path_vel, model_filename_vel)
#
# dim_input = 6
# dim_output = 1
# unit = 16
# layer = 1
# seq_len = final_data_train[1]
#
# model = create_model(dim_input, dim_output, unit, layer, seq_len)
# # Load the previously saved weights
# model.load_weights(filepath_vel)
# model.summary()
#
# seq_len = 1
# i = 0
# nn_input = tf.keras.layers.Input(shape=(seq_len, dim_input), batch_size=1)
# nn_mask = tf.keras.layers.Masking(mask_value=0)(nn_input)
# lstm_cell = tf.keras.experimental.PeepholeLSTMCell(unit)
# nn_cell, h, c = tf.keras.layers.RNN(lstm_cell, return_sequences=True, return_state=True, stateful=True,
#                                     name='LSTM{}'.format(i))(nn_mask)
# state = [h, c]
# nn_output = tf.keras.layers.Dense(dim_output)(nn_cell)
#
# model = tf.keras.Model(nn_input, [nn_output, state])
#
# model.summary()
#
# result = np.zeros((final_data_target.shape[1], final_data_target.shape[0]))
# state = np.zeros((final_data_train.shape[1], 2, final_data_train.shape[0], unit))
#
# start_time = time.perf_counter()
# # Run each timestep
# for i in range(final_data_target.shape[1]):
#     result1, state1 = model(final_data_train[:, i:i + 1, :])
#     result[i, :] = np.squeeze(result1)
#     h, c = state1
#     state[i, 0, :, :] = np.array(h)
#     state[i, 1, :, :] = np.array(c)
#
# print(result.shape)
# print(state.shape)