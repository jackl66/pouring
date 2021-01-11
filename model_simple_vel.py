'''
model class, base and derivatives
'''
import tensorflow as tf 
import define_variables
import numpy as np 

class BaseRNNModel:
    def __init__(self, num_unit, num_layer, dim_input, dim_output, scope_name):
        '''
        dropout_train: keep probability in training
        '''
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_unit = num_unit
        self.num_layer = num_layer
        # self.dropout_train = dropout_train
        self.scope_name = scope_name 

        self.keep_prob = tf.placeholder_with_default(1.0, (), name='keep_prob_{}'.format(self.scope_name))

        self.cell, self.variables, self.var_dict = define_variables.define_variables_advance(
            self.dim_input,
            self.dim_output,
            self.num_unit,
            self.num_layer,
            self.scope_name,
            self.keep_prob)

    
    def build(self, seq_len, batch_size=None):
        '''
        build the complete model which takes complete sequences as input
        '''
        # placeholder is abbreviated as "ph" in variable names
        self.x = tf.placeholder(tf.float32, [batch_size, seq_len, self.dim_input], name='x_{}'.format(self.scope_name))
        self.y = tf.placeholder(tf.float32, [batch_size, seq_len, self.dim_output], name='y_{}'.format(self.scope_name))
        self.mask = tf.placeholder(tf.float32, [batch_size, seq_len], name='mask_{}'.format(self.scope_name))
        self.lengths = tf.placeholder(tf.float32, [batch_size], name='lengths_{}'.format(self.scope_name))
        self.regularize_strength = tf.placeholder(tf.float32, (), name='regularize_strength_{}'.format(self.scope_name))

        # run
        with tf.variable_scope(self.scope_name, reuse=True):
            output, self.state = tf.nn.dynamic_rnn(self.cell,
                                            self.x,
                                            sequence_length=self.lengths,
                                            dtype=tf.float32)
        output = tf.reshape(output, [-1, self.num_unit])

        w = self.var_dict['w']
        b = self.var_dict['b']
        temp = tf.matmul(output, w) + b

        self.y_hat = tf.reshape(temp, [-1, seq_len, self.dim_output])  # estimated output
        #Added code to get the hidden states and cell state
        self.output = tf.reshape(output, [-1, seq_len, self.num_unit])
        #self.state = tf.reshape(self.state, [-1, seq_len, self.num_unit])

        diff_squared = tf.square(self.y - self.y_hat)
        loss = tf.reduce_mean(diff_squared, reduction_indices=2)
        loss *= self.mask
        loss = tf.reduce_sum(loss, reduction_indices=1)
        length_float = tf.cast(self.lengths, tf.float32)
        loss /= length_float
        self.loss = tf.reduce_mean(loss)

    def build_test(self, seq_len, batch_size):
        '''
        temporary

        build the complete model which takes complete sequences as input
        '''
        # placeholder is abbreviated as "ph" in variable names
        self.x = tf.placeholder(tf.float32, [batch_size, seq_len, self.dim_input], name='x_{}'.format(self.scope_name))
        self.y = tf.placeholder(tf.float32, [batch_size, seq_len, self.dim_output], name='y_{}'.format(self.scope_name))
        self.mask = tf.placeholder(tf.float32, [batch_size, seq_len], name='mask_{}'.format(self.scope_name))
        self.regularize_strength = tf.placeholder(tf.float32, (), name='regularize_strength_{}'.format(self.scope_name))
        self.state_array = tf.placeholder(tf.float32, [batch_size, self.num_layer, 2, self.num_unit], name="state_array_{}".format(self.scope_name))

        list_init_state = []
        for layer in range(self.num_layer):
            c = tf.slice(self.state_array, [0, layer, 0, 0], [batch_size, 1, 1, self.num_unit])
            c = tf.squeeze(c)
            h = tf.slice(self.state_array, [0, layer, 1, 0], [batch_size, 1, 1, self.num_unit])
            h = tf.squeeze(h)
            list_init_state.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
        init_state_feed = tuple(list_init_state)


        # run
        with tf.variable_scope(self.scope_name, reuse=True):
            output, self.state = tf.nn.dynamic_rnn(self.cell,
                                            self.x,
                                            dtype=tf.float32,
                                            initial_state=init_state_feed)
        output_reshaped = tf.reshape(output, [-1, self.num_unit])
        self.output = output
        w = self.var_dict['w']
        b = self.var_dict['b']
        temp = tf.matmul(output_reshaped, w) + b

        self.y_hat = tf.reshape(temp, [-1, seq_len, self.dim_output])  # estimated output


    def build_step(self, num_sample):
        '''
        run one step only
        '''

        self.x_step = tf.placeholder(tf.float32, [num_sample, self.dim_input], name="x_step_{}".format(self.scope_name))
        self.ch = tf.placeholder(tf.float32, [num_sample, self.num_layer, 2, self.num_unit], name="ch_{}".format(self.scope_name))

        list_init_state = []
        for layer in range(self.num_layer):
            c = tf.slice(self.ch, [0, layer, 0, 0], [num_sample, 1, 1, self.num_unit])
            c = tf.squeeze(c)
            h = tf.slice(self.ch, [0, layer, 1, 0], [num_sample, 1, 1, self.num_unit])
            h = tf.squeeze(h)
            list_init_state.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
        init_state_feed = tuple(list_init_state)

        w = self.var_dict['w']
        b = self.var_dict['b']

        with tf.variable_scope(self.scope_name, reuse=True) as vs:
            # with tf.variable_scope('RNN'):
            # use the initial state from feed_dict
            output, self.state_step = self.cell(self.x_step, init_state_feed)
        self.output_step = output
        self.y_hat_step = tf.matmul(output, w) + b


       
