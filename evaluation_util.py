import numpy as np
import matplotlib.pyplot as pp
import os

class ModelLoader():
    '''
    helps to load a model from checkpoint files
    '''
    def __init__(self, unit=None, layer=None, lr_string=None, epoch_total=None, epoch_load=None, prepend=None, seed=None, filepath=None):
        ''' 
        unit: number of units per layer
        layer: number of layers
        ls_string: string of learning rate 
        epoch_total: total number of epochs for training
        epoch_load: epoch to load
        prepend: prepend length
        seed: seed for random number generators 
        filepath: absolute path of model file

        It is better to set the attributes using this method than directly
        accessing the attributes. 
        '''
        self.unit = unit
        self.layer = layer
        self.lr_string = lr_string
        self.epoch_total = epoch_total
        self.epoch_load = epoch_load
        self.prepend = prepend
        self.seed = seed
        self.filepath = filepath

        # mean and std calculated from training data 
        self.mean_train = None # mean of training data
        self.std_train = None # standard deviation of the training data 



def find_lowest_cost_model(model_dir_path, is_train=True):
    '''
    input:
    model_dir_path: absolute path of model directory
    mode: whether to find the lowest training loss or validation loss
        if True, training loss, if False, validation loss
    return: 
    absolute path of checkpoint of best model
    '''
    files = os.listdir(model_dir_path)
    filename_best = ''
    c_best = 1e6
    for f in files:
        if f.endswith('index'):
            start_c = f.find('_c') + 2
            end_c = f.find('_cv')
            start_cv = end_c + 3
            end_cv = f.find('.ckpt')
            c = float(f[start_c : end_c])
            cv = float(f[start_cv : end_cv])
            x_meta = f.find('.index')
            c_current = c if is_train else cv
            if c_current < c_best:
                filename_best = f[:x_meta]
                c_best = c_current
    return filename_best

if __name__ == '__main__':
    model_path = '/Users/yongqiang/Documents/pouring_code_new/effect_driven_08_15_2018/train_desire_e3000lr0.01seed1534531981unit100layer1prepend60'
    print('train', find_lowest_cost_model(model_path))
    print('valid', find_lowest_cost_model(model_path, is_train=False))


