# Created by moritz (wolter@cs.uni-bonn.de) at 11/12/2019

import subprocess
import datetime
subprocess.call('pwd')

print('baseline GRU')
# time_str = str(datetime.datetime.today())
# print('adding problem baseline', time_str)
# with open("runs/v3_baselineGRU_adding_" + time_str + ".txt", "w") as f:
#     subprocess.call(['python', 'adding_memory_RNN_compression.py',
#                     '--problem', 'adding', '--cell', 'GRU'], stdout=f)
# time_str = str(datetime.datetime.today())
# print('memory problem baseline', time_str)
# with open("runs/v3_baselineGRU_memory_" + time_str + ".txt", "w") as f:
#     subprocess.call(['python', 'adding_memory_RNN_compression.py',
#                      '--problem', 'memory', '--cell', 'GRU'], stdout=f)

experiment_lst = ['state_reset', 'state_update', 'gates', 'full',
                  'update', 'state', 'reset']
print('adding memory wavelet compression')
for experiment in experiment_lst:
    time_str = str(datetime.datetime.today())
    print('adding ', experiment, ' at time:', time_str)
    with open("runs/v3_adding_" + experiment + time_str + ".txt", "w") as f:
        subprocess.call(['python', 'adding_memory_RNN_compression.py',
                         '--cell', 'WaveletGRU',
                         '--compression_mode', experiment,
                         '--problem', 'adding'], stdout=f)
    time_str = str(datetime.datetime.today())
    print('memory ', experiment, ' at time:', time_str)
    with open("runs/v3_memory_" + experiment + time_str + ".txt", "w") as f:
        subprocess.call(['python', 'adding_memory_RNN_compression.py',
                         '--cell', 'WaveletGRU',
                         '--compression_mode', experiment,
                         '--problem', 'memory'], stdout=f)

print('fastfood full compression')
time_str = str(datetime.datetime.today())
print('adding problem full compression', time_str)
with open("runs/v3_fastfood_adding_problem_full_compression_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'adding',
                     '--cell', 'FastFoodGRU', '--compression_mode', 'full'], stdout=f)
time_str = str(datetime.datetime.today())
print('memory problem full compression', time_str)
with open("runs/v3_fastfood_memory_problem_full_compression_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'memory',
                     '--cell', 'FastFoodGRU', '--compression_mode', 'full'], stdout=f)