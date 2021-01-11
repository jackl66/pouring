import numpy as np
import os
import tensorflow as tf

def variable_length(data):
    '''
    compute the mask and multiple sequence lengths from a zero-padded 
    data tensor of size [batch_size, input_size]
    '''
    num_sequences, max_len, feature_dim = data.shape
    data = np.max(np.abs(data), axis=2) # remove the feature dimension
    mask = np.sign(data)
    lengths = np.sum(mask, axis=1)
    lengths_rnn = lengths.astype(np.int32)
    mask_cost = mask
    # for i in range(data.shape[0]):
    #     mask_cost[i, lengths_rnn[i] - 1] = 0
    
    return (mask_cost, lengths_rnn)

def find_best_model(model_dir, model_name, mode="cost"):
    '''
    :param model_dir: absolute path of directory
    model_name: string of model_name
    :param mode: is either "cost" or "accuracy"
    :return: absolute path of checkpoint of best model
    '''
    files = os.listdir(model_dir)
    prefix = "m_" + model_name
    filename_best = ''
    if mode is "cost":
        c_best = 1e6
        for f in files:
            if f.startswith(prefix) and f.endswith('meta'):
                x_c = f.find('_c')
                x_cv = f.find('_cv')
                c = float(f[x_c + 2:x_cv])
                x_meta = f.find('.meta')
                if c < c_best:
                    filename_best = f[:x_meta]
                    c_best = c
    elif mode is "accuracy":
        a_best = 0
        for f in files:
            if f.startswith(prefix) and f.endswith('meta'):
                x_a = f.find('_a')
                x_av = f.find('_av')
                a = float(f[x_a + 2:x_av])
                x_meta = f.find('.meta')
                if a > a_best:
                    filename_best = f[:x_meta]
                    a_best = a
    else:
        print("mode should be either cost or accuracy.")
        return -1
    return filename_best

def add_eps(x):
    '''
    add smallest floating point value eps to a non-negative array of probabilities
    so that no values are zeros
    :return:
    '''
    eps = np.finfo('float').eps
    mask = (np.sign(eps - x) + 1) / 2
    print('mask')
    print(mask)
    x += eps * mask
    return x

def gaussian_np(y, mu, sigma):
    '''
    :param y: [batch_size * max_len, num_mixture]
    :param mu:
    :param sigma:
    :return: matrix of gaussian probability
    '''
    error_message = 'parameters shapes dont match: y.shape={}, mu.shape={}, sigma.shape={}'.format(y.shape, mu.shape, sigma.shape)
    assert (y.shape == mu.shape == sigma.shape), error_message
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (y - mu) ** 2 / (sigma ** 2))

def get_mean_std(data_train):
    '''
    need to disregard the zeros that are manually added
    '''
    _, lengths = variable_length(data_train)
    data_size, _, feature_dim = data_train.shape
    mean_train = np.zeros(feature_dim)
    std_train = np.zeros(feature_dim)
    for d in range(feature_dim):
        degree_list = []
        for n in range(data_size):
            degree_list.append(data_train[n, :lengths[n], d])
        degree_array = np.concatenate(degree_list)
        mean_train[d] = np.mean(degree_array)
        std_train[d] = np.std(degree_array)

    return (mean_train, std_train)


def apply_mean_std(data, mean_train, std_train):
    _, lengths = variable_length(data)
    data_copy = np.copy(data)
    data_size, _, feature_dim = data.shape
    for d in range(feature_dim):
        for n in range(data_size):
            data_copy[n, :lengths[n], d] -= mean_train[d]
            data_copy[n, :lengths[n], d] /= std_train[d]

    return data_copy

def count_total_variables():
    '''
    count the number of total trainable variables
    to be used locally in a file
    :return:
    '''
    total_parameters = 0
    #iterating over all variables
    for variable in tf.trainable_variables():
        local_parameters=1
        shape = variable.get_shape()  #getting shape of a variable
        for i in shape:
            local_parameters*=i.value  #mutiplying dimension values
        total_parameters+=local_parameters
    return total_parameters