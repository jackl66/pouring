# define variables for training (and evaluation)
# modified the way of stacking multiple layers for TensorFlow r1.3
# 1/1/2017

from __future__ import print_function
import numpy as np 
import os, pickle
import matplotlib.pyplot as pp
import tensorflow as tf
import util 
import time
from tensorflow.python.ops import array_ops
from six import string_types


def define_variables(input_dim, output_dim, num_units, num_layers, variable_scope_name, keep_prob):
    '''
    generalized version of define_variables_frc(or define_variables_vel)
    define variables for prediction system. Defined structure is compatible with
    using dynamic_rnn() and with running rnn using steps.
    arg:
    num_units: number of units per layer
    '''
    assert (isinstance(input_dim, int) and input_dim > 0), \
        'arg[0]: input_dim must be integer and greated than zero'
    assert (isinstance(output_dim, int) and output_dim > 0), \
        'arg[1]: output_dim must be integer and greated than zero'
    assert (isinstance(num_units, int) and num_units > 0), \
        'arg[2]: num_units must be integer and greated than zero'
    assert (isinstance(num_layers, int) and num_layers > 0), \
        'arg[3]: num_layers must be integer and greated than zero'
    assert isinstance(variable_scope_name, string_types), \
        'arg[4]: variable_scope_name must be a string'

    with tf.variable_scope(variable_scope_name) as vs:
        # RNN cell
        def lstm_cell():
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            # cell = tf.nn.rnn_cell.LSTMCell(num_units, use_peepholes=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

        # parameters for linear layer
        # w = tf.get_variable('w', initializer=tf.random_uniform([num_units, output_dim]))
        # b = tf.get_variable('b', initializer=tf.random_uniform([output_dim]))
        w = tf.get_variable('w', shape=[num_units, output_dim], initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable('b', shape=[output_dim], initializer=tf.glorot_uniform_initializer())
        
        # w_init = tf.get_variable('w_init', initializer=tf.random_uniform([input_dim, num_units]))
        # b_init = tf.get_variable('b_init', initializer=tf.random_uniform([num_units]))

        # run a fake step just to instantiate the cell variables
        batch_size_fake = 1
        input_fake = tf.zeros((batch_size_fake, input_dim))
        # with tf.variable_scope('RNN'):
        state = cell.zero_state(batch_size_fake, dtype=tf.float32)
        _, _ = cell(input_fake, state)


    with tf.variable_scope(vs):
        variables = [v for v in tf.global_variables()
                               if v.name.startswith(vs.name)]

    # alternative way of getting variables
    # variables_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variable_scope_name) 

    # dict of variables that would be used later
    var_dict = {}
    var_dict['w'] = w
    var_dict['b'] = b
    # var_dict['w_init'] = w_init
    # var_dict['b_init'] = b_init
    return (cell, variables, var_dict)


def define_variables_advance(input_dim, output_dim, num_units, num_layers, variable_scope_name, keep_prob):
    '''
    generalized version of define_variables_frc(or define_variables_vel)
    define variables for prediction system. Defined structure is compatible with
    using dynamic_rnn() and with running rnn using steps.
    arg:
    num_units: number of units per layer

    use peephole and better initialization
    '''
    assert (isinstance(input_dim, int) and input_dim > 0), \
        'arg[0]: input_dim must be integer and greated than zero'
    assert (isinstance(output_dim, int) and output_dim > 0), \
        'arg[1]: output_dim must be integer and greated than zero'
    assert (isinstance(num_units, int) and num_units > 0), \
        'arg[2]: num_units must be integer and greated than zero'
    assert (isinstance(num_layers, int) and num_layers > 0), \
        'arg[3]: num_layers must be integer and greated than zero'
    assert isinstance(variable_scope_name, string_types), \
        'arg[4]: variable_scope_name must be a string'

    with tf.variable_scope(variable_scope_name) as vs:
        # RNN cell
        def lstm_cell():
            # cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            cell = tf.nn.rnn_cell.LSTMCell(num_units, use_peepholes=True)
            # cell = tf.nn.rnn_cell.LSTMCell(num_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

        # parameters for linear layer
        # w = tf.get_variable('w', initializer=tf.random_uniform([num_units, output_dim]))
        # b = tf.get_variable('b', initializer=tf.random_uniform([output_dim]))
        w = tf.get_variable('w', shape=[num_units, output_dim], initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable('b', shape=[output_dim], initializer=tf.glorot_uniform_initializer())

        # w_init = tf.get_variable('w_init', initializer=tf.random_uniform([input_dim, num_units]))
        # b_init = tf.get_variable('b_init', initializer=tf.random_uniform([num_units]))

        # run a fake step just to instantiate the cell variables
        batch_size_fake = 1
        input_fake = tf.zeros((batch_size_fake, input_dim))
        # with tf.variable_scope('RNN'):
        state = cell.zero_state(batch_size_fake, dtype=tf.float32)
        _, _ = cell(input_fake, state)

    with tf.variable_scope(vs):
        variables = [v for v in tf.global_variables()
                     if v.name.startswith(vs.name)]

    # alternative way of getting variables
    # variables_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variable_scope_name)

    # dict of variables that would be used later
    var_dict = {}
    var_dict['w'] = w
    var_dict['b'] = b
    # var_dict['w_init'] = w_init
    # var_dict['b_init'] = b_init
    return (cell, variables, var_dict)


def define_var_mdn(input_dim, num_mixture, num_units, num_layers, variable_scope_name, keep_prob):
    "mixture density network"
    assert (isinstance(input_dim, int) and input_dim > 0), \
        'arg[0]: input_dim must be integer and greated than zero'
    assert (isinstance(num_mixture, int) and num_mixture > 0), \
        'arg[1]: output_dim must be integer and greated than zero'
    assert (isinstance(num_units, int) and num_units > 0), \
        'arg[2]: num_units must be integer and greated than zero'
    assert (isinstance(num_layers, int) and num_layers > 0), \
        'arg[3]: num_layers must be integer and greated than zero'
    assert isinstance(variable_scope_name, string_types), \
        'arg[4]: variable_scope_name must be a string'

    with tf.variable_scope(variable_scope_name) as vs:
        # from rotation to force
        # RNN cell
        def lstm_cell():
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
        # parameters for linear layer
        output_dim = num_mixture * 3
        w = tf.get_variable('w', initializer=tf.random_uniform([num_units, output_dim]))
        b = tf.get_variable('b', initializer=tf.random_uniform([output_dim]))

        w_init = tf.get_variable('w_init', initializer=tf.random_uniform([input_dim, num_units]))
        b_init = tf.get_variable('b_init', initializer=tf.random_uniform([num_units]))

        # run a fake step just to instantiate the cell variables
        batch_size_fake = 1
        input_fake = tf.zeros((batch_size_fake, input_dim))
        # with tf.variable_scope('RNN'):
        state = cell.zero_state(batch_size_fake, dtype=tf.float32)
        _, _ = cell(input_fake, state)

    with tf.variable_scope(vs):
        variables = [v for v in tf.global_variables()
                               if v.name.startswith(vs.name)]

    # alternative way of getting variables
    # variables_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variable_scope_name)

    # dict of variables that would be used later
    var_dict = {}
    var_dict['w'] = w
    var_dict['b'] = b
    var_dict['w_init'] = w_init
    var_dict['b_init'] = b_init
    return (cell, variables, var_dict)

def define_variables_gru(input_dim, output_dim, num_units, num_layers, variable_scope_name, keep_prob):
    '''
    generalized version of define_variables_frc(or define_variables_vel)
    define variables for prediction system. Defined structure is compatible with
    using dynamic_rnn() and with running rnn using steps.
    arg:
    num_units: number of units per layer

    GRU cell for comparison
    '''
    assert (isinstance(input_dim, int) and input_dim > 0), \
        'arg[0]: input_dim must be integer and greated than zero'
    assert (isinstance(output_dim, int) and output_dim > 0), \
        'arg[1]: output_dim must be integer and greated than zero'
    assert (isinstance(num_units, int) and num_units > 0), \
        'arg[2]: num_units must be integer and greated than zero'
    assert (isinstance(num_layers, int) and num_layers > 0), \
        'arg[3]: num_layers must be integer and greated than zero'
    assert isinstance(variable_scope_name, string_types), \
        'arg[4]: variable_scope_name must be a string'

    with tf.variable_scope(variable_scope_name) as vs:
        # RNN cell
        def gru_cell():
            # cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            cell = tf.nn.rnn_cell.GRUCell(num_units)
            # cell = tf.nn.rnn_cell.LSTMCell(num_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell() for _ in range(num_layers)])

        # parameters for linear layer
        # w = tf.get_variable('w', initializer=tf.random_uniform([num_units, output_dim]))
        # b = tf.get_variable('b', initializer=tf.random_uniform([output_dim]))
        w = tf.get_variable('w', shape=[num_units, output_dim], initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable('b', shape=[output_dim], initializer=tf.glorot_uniform_initializer())

        # w_init = tf.get_variable('w_init', initializer=tf.random_uniform([input_dim, num_units]))
        # b_init = tf.get_variable('b_init', initializer=tf.random_uniform([num_units]))

        # run a fake step just to instantiate the cell variables
        batch_size_fake = 1
        input_fake = tf.zeros((batch_size_fake, input_dim))
        # with tf.variable_scope('RNN'):
        state = cell.zero_state(batch_size_fake, dtype=tf.float32)
        _, _ = cell(input_fake, state)

    with tf.variable_scope(vs):
        variables = [v for v in tf.global_variables()
                     if v.name.startswith(vs.name)]

    # alternative way of getting variables
    # variables_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variable_scope_name)

    # dict of variables that would be used later
    var_dict = {}
    var_dict['w'] = w
    var_dict['b'] = b
    # var_dict['w_init'] = w_init
    # var_dict['b_init'] = b_init
    return (cell, variables, var_dict)


if __name__ == '__main__':
    input_dim = 1
    output_dim = 1
    num_units = 16
    num_layer = 1
    keep_prob = 0.5
    variable_scope_name = 'vel'
    _, variables_vel, variables_2 = define_variables_gru(
        input_dim, 
        output_dim,
        num_units,
        num_layer,
        variable_scope_name,
        keep_prob)

    # saver = tf.train.Saver(variables_frc)
    saver = tf.train.Saver(variables_vel)
    saver = tf.train.Saver(variables_2)
    init_op = tf.global_variables_initializer()    
    with tf.Session() as sess:
        sess.run(init_op)
        # saver.save(sess, './def_var_frc.ckpt')
        saver.save(sess, './def_var_vel.ckpt')
        saver.save(sess, './def_var_vel_2.ckpt')
        # saver.restore(sess, './m_vel.ckpt')
        print('session ended.')
