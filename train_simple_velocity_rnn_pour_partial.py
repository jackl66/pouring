'''
derived from train_simple_predict_use_class.py
'''

import numpy as np
import os, time
import matplotlib.pyplot as pt
import tensorflow as tf
import util
import argparse as ap
import preprocessor
import model_simple_vel
import pickle
import evaluation_util

parser = ap.ArgumentParser()
# positional argument
parser.add_argument("unit", help="number of units per layer", type=int)
parser.add_argument("layer", help="number of layers", type=int)
parser.add_argument("lr", help="initial learning rate", type=float)
parser.add_argument("epoch", help="number of epochs of run", type=int)
parser.add_argument("prepend", help="length of prepending the data for stabilizing the system", type=int)
# optional argument
parser.add_argument("-regularize", help="regularization strength for sigma")
parser.add_argument("-gpu", help="index of gpu to run on, starts with 0")
parser.add_argument("-dropout", help="probability of keeping activations", type=float)
args = parser.parse_args()

print('\n')
print('unit', args.unit)
print('layer', args.layer)
print('lr', args.lr)
print('epoch', args.epoch)
print('prepend length', args.prepend)
print('gpu', args.gpu)
print('dropout', args.dropout)
print('regularize', args.regularize)
print('\n')

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

if args.gpu:
    os.environ["CUDA_DECIVE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

train_ratio = 0.7
valid_ratio = 0.2
#seed = int(time.time())
seed = 1599086380
preprocessed_data = preprocessor.SimpleVelPreprocessor('data_pour_partial.npy', seed, train_ratio, valid_ratio, prepend_len=args.prepend)


dim_input = 6
dim_output = 1
unit = args.unit
layer = args.layer
dropout = args.dropout if args.dropout else 0.0
regu_strength = args.regularize if args.regularize else 0.0
seq_len = preprocessed_data.input_train.shape[1]

model = create_model(dim_input,dim_output,unit,layer,seq_len,dropout)

model.summary()

wd = os.getcwd()
dirname = 'train_simple_vel_partial_e{}lr{}seed{}unit{}layer{}prepend{}dropout{}regu{}'.format(args.epoch,
                                                               args.lr,
                                                               seed,
                                                               args.unit,
                                                               args.layer,
                                                               args.prepend,
                                                               dropout,
                                                               regu_strength
                                                               )
# make sure directory exists
if not (os.path.isdir(dirname) and os.path.exists(dirname)):
    os.mkdir(dirname)


checkpoint_path = os.path.join(wd, dirname, "e{epoch:d}_model_c{loss:.4f}_cv{val_loss:.4f}.ckpt")

cp_callback = tf.keras.callbacks.ModelCheckpoint(
		    filepath=checkpoint_path, 
			    verbose=1, 
				    save_weights_only=True,
					    period=5)

log_file = os.path.join(wd, dirname,'training.log')

csv_logger = tf.keras.callbacks.CSVLogger(log_file)

model.fit(x=preprocessed_data.input_train,y=preprocessed_data.target_train,validation_data=(preprocessed_data.input_valid,preprocessed_data.target_valid),batch_size=256,epochs=args.epoch,verbose=1,callbacks=[cp_callback,csv_logger])


# Create a new model instance
model = create_model(dim_input,dim_output,unit,layer,seq_len)

wd = os.getcwd()
model_dir_path_vel = os.path.join(wd, dirname)
model_filename_vel = evaluation_util.find_lowest_cost_model(model_dir_path_vel, is_train=False)
filepath_vel = os.path.join(model_dir_path_vel, model_filename_vel)

# Load the previously saved weights
model.load_weights(filepath_vel)

result = model.predict(x=preprocessed_data.input_test)


indexes = [0,10,20,30]
i = 1
for idx in indexes:
	pt.subplot(2,2,i)
	i += 1
	pt.plot(preprocessed_data.target_test[idx,:,:],'b',label='Ground Truth')
	pt.plot(result[idx,:,:],'r',label='Prediction')

pt.show()

