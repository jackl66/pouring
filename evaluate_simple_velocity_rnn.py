import os
import tensorflow as tf
import numpy as np
import evaluation_util
import preprocessor
import model_simple_vel
import matplotlib.pyplot as pt
import util
import matplotlib.ticker as tk
import scipy.signal as sg


def create_model(dim_input,dim_output,unit,layer,seq_len,dropout=0):
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
	
	#Adds the input dimension and does not take into consideration zero values of the input sequences
	model.add(tf.keras.layers.Input(shape=(seq_len,dim_input)))
	model.add(tf.keras.layers.Masking(mask_value=0))
	
	#Adds the different rnn layers
	for i in range(layer):
		lstm_cell = tf.keras.experimental.PeepholeLSTMCell(unit) 
		model.add(tf.keras.layers.RNN(lstm_cell,return_sequences=True,name='LSTM{}'.format(i)))
		model.add(tf.keras.layers.Dropout(dropout))
	
	#Adds the output dense layer
	model.add(tf.keras.layers.Dense(dim_output,activation='linear'))
	
	#Defines the loss and optimizer with default values
	model.compile(loss='mean_squared_error', optimizer='adam')
	
	return model

wd = os.getcwd()
model_dir_path_vel = os.path.join(wd, 'train_simple_vel_partial_e2000lr0.001seed1599086380unit16layer1prepend60dropout0.5regu0.0')

model_filename_vel = evaluation_util.find_lowest_cost_model(model_dir_path_vel, is_train=False)
filepath_vel = os.path.join(model_dir_path_vel, model_filename_vel)

train_ratio = 0.7
valid_ratio = 0.2
prepend_length = 60
seed = 1599086380
preprocessed_data = preprocessor.SimpleVelPreprocessor('data_pour_partial.npy', seed, train_ratio, valid_ratio, prepend_len=prepend_length)

dim_input = 6
dim_output = 1
unit = 16
layer = 1
seq_len = preprocessed_data.input_train.shape[1]

model = create_model(dim_input,dim_output,unit,layer,seq_len)

# Load the previously saved weights
model.load_weights(filepath_vel)

result = model.predict(x=preprocessed_data.input_test)
lengths = preprocessed_data.lengths_test


indexes = [4,8,12,16]
i = 1
for idx in indexes:
	pt.subplot(2,2,i)
	i += 1
	pt.plot(preprocessed_data.target_test[idx,0:lengths[idx],:],'b',label='Ground Truth')
	pt.plot(result[idx,0:lengths[idx],:],'r',label='Prediction')

pt.show()

