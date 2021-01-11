'''
code that does preprocessing ONLY
intended to be used by multiple files 
'''

import matplotlib.pyplot as pp 
import numpy as np 
import copy
import util
import scipy.ndimage

def prepend_data(x, prepend_length):
    '''
    prepend certain amount of data for system stablization
    assumes data is of shape [data_size, max_len, feature_dim]
    We just copy the first time stamp to all the prepended part
    '''
    x_2_prepend = np.tile(x[:, 0:1, :], (1, prepend_length, 1))
    x_prepended = np.concatenate((x_2_prepend, x), axis=1)
    # print('new shape', x_prepended.shape)
    #
    # for d in range(x.shape[2]):
    #     pp.subplot(6, 1, d+1)
    #     pp.plot(x[0, :, d])
    # pp.show()
    # for d in range(x.shape[2]):
    #     pp.subplot(6, 1, d+1)
    #     pp.plot(x_prepended[0, :, d])
    # pp.show()
    return x_prepended

def adjust_mask_and_lengths(mask_orig, lengths_orig, prepend_length):
    '''
    adjust the mask and lengths for prepended data
    :return:
    '''
    mask_to_prepend = np.zeros((mask_orig.shape[0], prepend_length), dtype=np.int32)
    mask_prepended = np.concatenate((mask_to_prepend, mask_orig), axis=1)
    lengths_prepended = lengths_orig + prepend_length

    return (mask_prepended, lengths_prepended)

def get_real_data(x):
    '''
    get actual data from x, which does not include all the manually added zeros
    returns a list, each element is a numpy, corresponding to a single degree
    '''
    _, lengths = util.variable_length(x)
    data_size, _, feature_dims = x.shape
    real_data_list = []
    for d in range(feature_dims):
        d_list = []
        for n in range(data_size):
            d_list.append(copy.deepcopy(x[n, :lengths[n], d]))
        real_data_list.append(np.concatenate(d_list))
    return real_data_list


def plot_histogram(data):
    data_real = get_real_data(data)
    for d in range(data.shape[2]):
        pp.subplot(data.shape[2], 1, d+1)
        hist_values, bin_edges = np.histogram(data[:, :, d], bins='auto')
        pp.hist(data_real[d], bins='auto')
    pp.show()


class BasePreprocessor:
    '''
    shared procedures for data preprocessing
    ''' 
    def __init__(self, data_name, seed, train_ratio, valid_ratio, prepend_len=None):
        data = np.load(data_name)
        np.random.seed(seed)
        np.random.shuffle(data)
        data_size, max_len, feature_dim = data.shape

        self.data = data
        _, self.lengths = util.variable_length(data) 
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.prepend_len = prepend_len

        # prepare input and target data
        self._design_input_target()

        # split into train, valid and test, normalize
        self._process_data()
        # prepend_data with normalization
        if self.prepend_len:
            self._prepend(self.prepend_len)

    def _design_input_target(self):
        self.data_input = None
        self.data_target = None

    def _process_data(self):
        '''
        1. split data into train, valid, and test
        2. normalize the train, valid, and test using the train data
        '''
        data_input = self.data_input
        data_target = self.data_target
        train_ratio = self.train_ratio
        valid_ratio = self.valid_ratio
        # todo split the data into three parts: train, validation, test
        data_size = data_input.shape[0]
        data_boundary_train = int(train_ratio * data_size)
        data_boundary_valid = int((train_ratio + valid_ratio) * data_size)
        data_input_train = data_input[:data_boundary_train, :, :]
        data_input_valid = data_input[data_boundary_train:data_boundary_valid, :, :]
        data_input_test = data_input[data_boundary_valid:, :, :]
        # todo what is the difference between data_input and data_target, data_target=ground truth?
        data_target_train = data_target[:data_boundary_train, :, :]
        data_target_valid = data_target[data_boundary_train:data_boundary_valid, :, :]
        data_target_test = data_target[data_boundary_valid:, :, :]

        # the mask and lengths are determined by the data, so
        # appending the zeros correctly
        data_mask_train, data_lengths_train = util.variable_length(data_input_train)
        data_mask_valid, data_lengths_valid = util.variable_length(data_input_valid)
        data_mask_test, data_lengths_test = util.variable_length(data_input_test)

        self.mean_train, self.std_train = util.get_mean_std(data_input_train)
        print('before...')
        print('mean train', self.mean_train)
        print('std train', self.std_train)
        data_input_train = util.apply_mean_std(data_input_train, self.mean_train, self.std_train)  # standardize training data
        mean_train_new, std_train_new = util.get_mean_std(data_input_train)
        print('after...')
        print('mean train', mean_train_new)
        print('std train', std_train_new)

        data_input_valid = util.apply_mean_std(data_input_valid, self.mean_train, self.std_train)
        mean_valid, std_valid = util.get_mean_std(data_input_valid)
        print('valid...')
        print('mean valid', mean_valid)
        print('std valid', std_valid)

        data_input_test = util.apply_mean_std(data_input_test, self.mean_train, self.std_train)
        mean_test, std_test = util.get_mean_std(data_input_test)
        print('test...')
        print('mean test', mean_test)
        print('std test', std_test)

        # the data have been normalized 
        self.input_train = data_input_train
        self.input_valid = data_input_valid
        self.input_test = data_input_test
        self.target_train = data_target_train
        self.target_valid = data_target_valid
        self.target_test = data_target_test
        self.mask_train = data_mask_train
        self.mask_valid = data_mask_valid
        self.mask_test = data_mask_test
        self.lengths_train = data_lengths_train
        self.lengths_valid = data_lengths_valid
        self.lengths_test = data_lengths_test

    
    def _prepend(self, prepend_length):
        '''
        If this function is called, it must be called as the last step of preprocessing
        :param prepend_length:
        :return:
        '''
        # adjust to new prepend length
        self.mask_train, self.lengths_train = adjust_mask_and_lengths(self.mask_train, self.lengths_train, prepend_length)
        self.mask_valid, self.lengths_valid = adjust_mask_and_lengths(self.mask_valid, self.lengths_valid, prepend_length)
        self.mask_test, self.lengths_test = adjust_mask_and_lengths(self.mask_test, self.lengths_test, prepend_length)

        self.input_train = prepend_data(self.input_train, prepend_length)
        self.target_train = prepend_data(self.target_train, prepend_length)
        self.input_valid = prepend_data(self.input_valid, prepend_length)
        self.target_valid = prepend_data(self.target_valid, prepend_length)
        self.input_test = prepend_data(self.input_test, prepend_length)
        self.target_test = prepend_data(self.target_test, prepend_length)

class DesirePreprocessor(BasePreprocessor):
    '''
    preprocessor class for the desired effect generation engine
    '''

    def _design_input_target(self):
        data = self.data
        data_lengths = self.lengths
        data_size, max_len, _ = self.data.shape

        data_ft = data[:, :, 1:2]
        data_fpourable = data[:, :, 2:3]
        data_f2pour = data[:, :, 3:4]
        data_height = data[:, :, 4:5]
        data_diameter = data[:, :, 5:6]
        data_ft_smooth = data[:, :, 6:7]

        data_progress = np.zeros([data_size, max_len, 1])
        for n in range(data_size):
            data_progress[n, :data_lengths[n], :] = -data_ft[n, :data_lengths[n], :] / data_f2pour[n, :data_lengths[n], :]

        # smoothed version of the progress
        data_progress_smooth = np.zeros([data_size, max_len, 1])
        for n in range(data_size):
            data_progress_smooth[n, :data_lengths[n], :] = -data_ft_smooth[n, :data_lengths[n], :] / data_f2pour[n, :data_lengths[n],
                                                                                       :]
        # compute progress increment
        data_progress_increment = np.zeros(data_progress.shape)
        for n in range(data_size):
            data_progress_increment[n, :data_lengths[n]-1] = data_progress_smooth[n, 1:data_lengths[n]] - data_progress_smooth[n, :data_lengths[n]-1]
            data_progress_increment[n, data_lengths[n]-1] = data_progress_increment[n, data_lengths[n]-2]

        data_curvature = np.zeros([data_size, max_len, 1])
        for n in range(data_size):
            data_curvature[n, :data_lengths[n], :] = 2 / data_diameter[n, :data_lengths[n], :]


        data_input = np.concatenate((data_progress,
                                    data_fpourable,
                                    data_height,
                                    data_curvature), axis=2)
        data_target = data_progress_increment


        self.data_input = data_input
        self.data_target = data_target
# todo
class SimpleVelPreprocessor(BasePreprocessor):
    '''
    preprocessor class for generating velocity
    '''

    def _design_input_target(self):
        data = self.data
        data_lengths = self.lengths
        data_size, max_len, _ = self.data.shape

        data_angle = data[:, :, 0:1]
        data_ft = data[:, :, 1:2]
        data_fpourable = data[:, :, 2:3]
        data_f2pour = data[:, :, 3:4]
        data_height = data[:, :, 4:5]
        data_diameter = data[:, :, 5:6]
        data_ft_smooth = data[:, :, 6:7]

        # compute velocity
        velocity = np.zeros(data_angle.shape)
        for n in range(data_size):
            velocity[n, :data_lengths[n]-1] = data_angle[n, 1:data_lengths[n]] - data_angle[n, :data_lengths[n]-1]
            velocity[n, data_lengths[n]-1] = velocity[n, data_lengths[n]-2]
        #  ground truth
        velocity *= 60 # 60Hz in the data, velocity now in degrees per second
        velocity *= np.pi / 180 # velocity now in radians per second

        data_curvature = np.zeros([data_size, max_len, 1])
        for n in range(data_size):
            data_curvature[n, :data_lengths[n], :] = 2 / data_diameter[n, :data_lengths[n], :]

        data_input = np.concatenate((data_angle,
                                     data_ft,
                                     #data_ft_smooth, #Put back data_ft
                                     data_fpourable,
                                     data_f2pour,
                                     data_height,
                                     data_curvature), axis=2)
        data_target = velocity  #Put back velocity

        #data_target = data_ft
        self.data_input = data_input
        self.data_target = data_target

class PredictPreprocessor(BasePreprocessor):
    def _design_input_target(self):
        data = self.data
        data_lengths = self.lengths
        data_size, max_len, _ = self.data.shape
        
        data_angle = data[:, :, 0:1]
        data_ft = data[:, :, 1:2]
        data_fpourable = data[:, :, 2:3]
        data_height = data[:, :, 4:5]
        data_diameter = data[:, :, 5:6]
        data_ft_smooth = data[:, :, 6:7]

        data_curvature = np.zeros([data_size, max_len, 1])
        for n in range(data_size):
            data_curvature[n, :data_lengths[n], :] = 2 / data_diameter[n, :data_lengths[n], :]

        # angular velocity
        data_vel = np.zeros([data_size, max_len, 1])
        for n in range(data_size):
            data_vel[n, :data_lengths[n]-1, 0] = data[n, 1:data_lengths[n], 0] - data[n, :data_lengths[n]-1, 0]
            data_vel[n, data_lengths[n]-1, 0] = data_vel[n, data_lengths[n]-2, 0]
        fs = 60 # sampling frequency: 60 Hz
        data_vel *= fs # absolute angular velocity

        # force increment
        data_ft_increment = np.zeros([data_size, max_len, 1])
        for n in range(data_size):
            data_ft_increment[n, :data_lengths[n]-1, :] = data_ft_smooth[n, 1:data_lengths[n], :] - data_ft_smooth[n, :data_lengths[n]-1, :]
            data_ft_increment[n, data_lengths[n]-1, 0] = data_ft_increment[n, data_lengths[n]-2, 0]

        # force left
        data_f_left = np.zeros([data_size, max_len, 1])
        for n in range(data_size):
            data_f_left[n, :data_lengths[n], :] = data_fpourable[n, :data_lengths[n], :] + data_ft[n, :data_lengths[n], :]


        data_input = np.concatenate((data_angle,
                                    data_ft,
                                    data_f_left,
                                    data_vel,
                                    data_fpourable,
                                    data_height,
                                    data_curvature), axis=2)
        data_target = data_ft_increment

        self.data_input = data_input
        self.data_target = data_target


class SimplePredictPreprocessor(BasePreprocessor):
    def _design_input_target(self):
        data = self.data
        data_lengths = self.lengths
        data_size, max_len, _ = self.data.shape

        data_angle = data[:, :, 0:1]
        data_ft = data[:, :, 1:2]
        data_fpourable = data[:, :, 2:3]
        data_height = data[:, :, 4:5]
        data_diameter = data[:, :, 5:6]
        data_ft_smooth = data[:, :, 6:7]

        data_curvature = np.zeros([data_size, max_len, 1])
        for n in range(data_size):
            data_curvature[n, :data_lengths[n], :] = 2 / data_diameter[n, :data_lengths[n], :]

        # angle starting with the second time stamp, i.e., left shifted by one
        data_angle_future = np.zeros([data_size, max_len, 1])
        for n in range(data_size):
            data_angle_future[n, :data_lengths[n]-1, :] = data_angle[n, 1:data_lengths[n], :]

        # nullify the last force by making it zero
        for n in range(data_size):
            data_ft[n, data_lengths[n]-1, :] = 0

        # nullify the last height by making it zero
        for n in range(data_size):
            data_height[n, data_lengths[n]-1, :] = 0

        # nullify the last curvature by making it zero
        for n in range(data_size):
            data_curvature[n, data_lengths[n]-1, :] = 0

        # nullify the last fpourable by making it zero
        for n in range(data_size):
            data_fpourable[n, data_lengths[n]-1, :] = 0

        # similar to angle_future, starts with the second time stamp, left shifted by one
        data_ft_smooth_future = np.zeros([data_size, max_len, 1])
        for n in range(data_size):
            data_ft_smooth_future[n, :data_lengths[n] - 1, :] = data_ft_smooth[n, 1:data_lengths[n], :]

        data_input = np.concatenate((data_angle_future,
                                     data_ft,
                                     data_fpourable,
                                     data_height,
                                     data_curvature), axis=2)
        data_target = data_ft_smooth_future

        self.data_input = data_input
        self.data_target = data_target


if __name__ == '__main__':
    # preprocessor_desire = DesirePreprocessor(101209, 0.7, 0.2)
    # preprocessor_predict = PredictPreprocessor(101209, 0.7, 0.2)
    # p = DesirePreprocessor('data_for_desire.npy', 101209, 0.7, 0.2, prepend_len=60)
    p = PredictPreprocessor('data_for_force_prediction.npy', 101209, 0.7, 0.2, prepend_len=60)
    # mask_train = p.mask_train
    length_train = p.lengths_train
    print(length_train)


    p_short = SimplePredictPreprocessor('data_for_force_prediction.npy', 101209, 0.7, 0.2, prepend_len=60)
    mask_train = p.mask_train
    length_train_short = p.lengths_train
    print(length_train_short)

    print('difference')
    print(p.lengths_train - p_short.lengths_train)

