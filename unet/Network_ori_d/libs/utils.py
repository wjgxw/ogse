import os
import sys
from os.path import join
import tensorflow as tf
import numpy as np
import random
import scipy.io as matio
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul

def get_gpu_id(gpu_num = 1, total_gpu = 6):
    command = 'nvidia-smi'
    info = os.popen(command).readlines()
    for message in info:
        sys.stdout.write(message)

    basic_info = info[8: (8+total_gpu*3): 3]
    index = range(len(basic_info))
    memory = list(map(lambda x: x[35:40], basic_info))
    util_ratio = list(map(lambda x: x[60:63], basic_info))

    details_info = info[(12+3*total_gpu):-1]
    process_name = list(map(lambda x: x[24:69], details_info))
    gpu_index = list(map(lambda x: x[4:6], details_info))
    pid = list(map(lambda x: x[11:16], details_info))
    occupy = list(map(lambda x: x[-11:-6], details_info))

    users = []
    for idx in pid:
        search_user_command = 'ps -ef | grep ' + str(idx)
        user_details = os.popen(search_user_command).readlines()
        user = list(filter(
            lambda x: (
                ('grep' not in x) and (int(x[9:14]) == int(idx))
            ),
            user_details
        ))
        if user:
            user = user.pop()[:8]
            users.append(user)
        else:
            pass

    zip_details = zip(users, gpu_index, occupy, pid, process_name)

    print('users     gpu_idx  occupy   pid       process_name')
    for ii in sorted(zip_details, key = lambda x: x[0]):
        print('    '.join(list(map(str, ii))))

    print(' ^-^ Current GPU memory list:')
    sys.stdout.write(' ^-^ ')
    print(list(map(int, memory)))

    sort_memory = sorted(zip(index, list(map(int, memory))), key = lambda x: x[1])
    GPU_ID = ','.join(
        list(map(lambda x: str(x[0]), sort_memory[:gpu_num]))
    )
    slect_memory = list(map(lambda x: int(x[1]), sort_memory[:gpu_num]))

    print(' ^-^ It is GPU %s \'s pleasure to serve you' % (GPU_ID))
    sys.stdout.write(' ^-^ Our memory is: ')
    print(slect_memory)

    return GPU_ID

def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul,[dim.value for dim in shape],1)
    return num_params
