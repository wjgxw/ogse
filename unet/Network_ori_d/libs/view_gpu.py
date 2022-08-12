# -*- coding: utf-8 -*-

import os
import sys

gpu_num = 4
command = 'nvidia-smi'
info = os.popen(command).readlines()
for message in info:
    sys.stdout.write(message)

basic_info = info[8:18:3]
index = range(len(basic_info))
memory = map(lambda x: x[35:40], basic_info)
util_ratio = map(lambda x: x[60:63], basic_info)

details_info = info[24:-1]
process_name = map(lambda x: x[24:70], details_info)
gpu_index = map(lambda x: x[4:6], details_info)
pid = map(lambda x: x[11:16], details_info)
occupy = map(lambda x: x[-11:-6], details_info)

users = []
for idx in pid:
    search_user_command = 'ps -ef | grep ' + str(idx)
    user_details = os.popen(search_user_command).readlines()
    user = filter(lambda x: 'grep' not in x, user_details)
    if user:
        user = user.pop()[:8]
        users.append(user)
    else:
        pass

zip_details = zip(users, gpu_index, occupy, pid, process_name)
print('users     gpu_idx  occupy   pid     process_name')
for ii in sorted(zip_details, key = lambda x: x[0]):
    print('    '.join(map(str, ii)))

print(' ^-^ Current GPU memory list:')
sys.stdout.write(' ^-^ ')
print(map(int, memory))

sort_memory = sorted(zip(index, map(int, memory)), key = lambda x: x[1])
GPU_ID = ','.join(
    map(lambda x: str(x[0]), sort_memory[:gpu_num])
)
slect_memory = map(lambda x: int(x[1]), sort_memory[:gpu_num])

print(' ^-^ It is GPU %s \'s pleasure to serve you' % (GPU_ID))
sys.stdout.write(' ^-^ Our memory is: ')
print(slect_memory)


















