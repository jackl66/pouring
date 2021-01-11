#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Juan Wilches
#
# Distributed under terms of the MIT license.

import tensorflow as tf
#from tensorflow.python.ops.rnn_cell import RNNCell
#from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers import AbstractRNNCell
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops

class SCell(AbstractRNNCell):

	'''
	Shadow Unit for recurrent neural networks.
	'''

	def __init__(self, units,input_size, **kwargs):
		self.units = units
		self.input_size = input_size
		super(SCell, self).__init__(**kwargs)
	
	@property
	def output_size(self):
		return self.units
	@property
	def state_size(self): #Definition of the state size
		return [self.units,self.units, self.input_size]
	

	def build(self, input_shape):
		
		self.kernel = self.add_weight(shape=(input_shape[-1], self.units*3),
									  initializer='glorot_uniform',
									  name='kernel')
		
		self.recurrent_kernel = self.add_weight(
			shape=(self.units, self.units*3),
			initializer='glorot_uniform',
			name='recurrent_kernel')

		self.bias = self.add_weight(
			shape=(self.units*3,),
			initializer='zeros',
			name='bias')


		self.i_peephole_weights = self.add_weight(
			shape=(self.units,),
			name='i_peephole_weights',
			initializer='glorot_uniform')
		self.g_peephole_weights = self.add_weight(
			shape=(self.units,),
			name='g_peephole_weights',
			initializer='glorot_uniform')
		self.h_peephole_weights = self.add_weight(
			shape=(self.units,),
			name='h_peephole_weights',
			initializer='glorot_uniform')

		self.built = True

	def call(self, inputs, states):
		
		s_prev = states[0]
		h_prev = states[1]
		i_prev = states[2]
		
		k_i, k_g, k_h = array_ops.split(self.kernel, num_or_size_splits=3, axis=1)
		b_i, b_g, b_h = array_ops.split(self.bias, num_or_size_splits=3, axis=0)

		x_i = K.dot(i_prev, k_i)
		x_i = K.bias_add(x_i,b_i)
		h_i = K.dot(h_prev, self.recurrent_kernel[:, :self.units]) 
		p_i = self.i_peephole_weights * s_prev
		i = tf.sigmoid(x_i + h_i + p_i)

		x_g = K.dot(i_prev, k_g)
		x_g = K.bias_add(x_g,b_g)
		h_g = K.dot(h_prev, self.recurrent_kernel[:, self.units:self.units*2]) 
		p_g = self.g_peephole_weights * s_prev
		g = tf.tanh(x_g + h_g + p_g)

		s = s_prev + i * g
		
		x_h = K.dot(inputs, k_h) 
		x_h = K.bias_add(x_h,b_h)
		h_h = K.dot(h_prev, self.recurrent_kernel[:, self.units*2:self.units*3]) 
		p_h = self.h_peephole_weights * tf.tanh(s)
		h = tf.tanh(x_h + h_h + p_h)
		
		return h, [s, h, inputs]

'''
	def __init__(self, num_units, input_size=None):
		self._num_units = num_units
		self._input_size = num_units if input_size is None else input_size
		super(SCell, self).__init__(**kwargs)

	@property
	def input_size(self):
		return self._input_size

	@property
	def output_size(self):
		return self._num_units

	@property
	def state_size(self): #Definition of the state size
		return self._num_units*2 + self._input_size
	

	def build(self, input_shape):
		self.kernel = self.add_weight(shape=(input_shape[-1], self.units),initializer='uniform',name='kernel')
		self.recurrent_kernel = self.add_weight(


	def call(self, inputs, state):
		
		#Shadow Unit with nunits cells.
		
		s_prev, h_prev, i_prev = array_ops.split(value=state, num_or_size_splits=[self._num_units,self._num_units,self._input_size], axis=1)
		
		self.kernel = self.add_weight
		with tf.variable_scope(scope or type(self).__name__):  # Shadow variable RNN
			with tf.variable_scope("Gates"):
				i = linear([i_prev, h_prev], self._num_units, True, 1.0, scope='Recurrent')
				p = linear([s_prev], self._num_units, False,scope='Peephole', hadamard=True)
				i = tf.sigmoid(i + p)
			with tf.variable_scope("Candidate"):
				g = linear([i_prev, h_prev], self._num_units,True,scope='Recurrent')
				p = linear([s_prev], self._num_units, False,scope='Peephole', hadamard=True)
				g = tf.tanh(g + p)
			s = s_prev + i * g
			with tf.variable_scope("Hidden"):
				h = linear([inputs, h_prev], self._num_units,True,scope='Recurrent')
				p = linear([tf.tanh(s)], self._num_units, False,scope='Peephole', hadamard=True)
				h = tf.tanh(h + p)
			new_state = array_ops.concat([s, h, inputs], 1)
		return h, new_state
		

def linear(args, output_size, bias, bias_start=0.0, scope=None, hadamard=False):
	"""Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
	Args:
	  args: a 2D Tensor or a list of 2D, batch x n, Tensors.
	  output_size: int, second dimension of W[i].
	  bias: boolean, whether to add a bias term or not.
	  bias_start: starting value to initialize the bias; 0 by default.
	  scope: VariableScope for the created subgraph; defaults to "Linear".

	Returns:
	  A 2D Tensor with shape [batch x output_size] equal to
	  sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

	Raises:
	  ValueError: if some of the arguments has unspecified or wrong shape.
	"""
	if args is None or (isinstance(args, (list, tuple)) and not args):
		raise ValueError("`args` must be specified")
	if not isinstance(args, (list, tuple)):
		args = [args]

	# Calculate the total size of arguments on dimension 1.
	total_arg_size = 0
	shapes = [a.get_shape().as_list() for a in args]
	for shape in shapes:
		if len(shape) != 2:
			raise ValueError(
				"Linear is expecting 2D arguments: %s" % str(shapes))
		if not shape[1]:
			raise ValueError(
				"Linear expects shape[1] of arguments: %s" % str(shapes))
		else:
			total_arg_size += shape[1]

	# Now the computation.
	with tf.variable_scope(scope or "Linear"):
		if hadamard == False:
			matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
		else:
			matrix = tf.get_variable("Matrix", [output_size])
		if len(args) == 1:
			if hadamard == False:
				res = tf.matmul(args[0], matrix)
			else:
				res = args[0] * matrix
		else:
			res = tf.matmul(tf.concat(args,1), matrix)
		if not bias:
			return res
		bias_term = tf.get_variable(
			"Bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
	return res + bias_term
'''
