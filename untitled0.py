# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 15:22:56 2020

@author: jackl
"""

import numpy as np

arr=np.zeros(6)
arr=[[2.57054564e+01, 1.03371712e-01, 2.65721940e-01, 1.20684726e-01,
 6.36882466e+01, 4.93812157e-03]]

print(np.shape(arr))

arr=np.squeeze(arr)
print(np.shape(arr))