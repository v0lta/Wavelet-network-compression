# Created by moritz (wolter@cs.uni-bonn.de) at 11/12/2019

import subprocess
import datetime
subprocess.call('pwd')

print('baseline GRU')
print('adding problem baseline')
time_str = str(datetime.datetime.today())
with open("runs/baselineGRU_adding_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'adding'], stdout=f)
print('memory problem baseline')
time_str = str(datetime.datetime.today())
with open("runs/baselineGRU_memory_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'memory'], stdout=f)

experiment_lst = ['state', 'update', 'reset', 'state_reset', 'state_update', 'gates', 'full']
print('adding memory wavelet compression')
for experiment in experiment_lst:
    time_str = str(datetime.datetime.today())
    print('adding ', experiment, ' at time:', time_str)
    with open("runs/adding_" + experiment + "_compression_seq_mnist_" + time_str + ".txt", "w") as f:
        subprocess.call(['python', 'adding_memory_RNN_compression.py',
                         '--cell', 'WaveletGRU', '--compression_mode', experiment,
                         '--problem', 'adding'], stdout=f)
    time_str = str(datetime.datetime.today())
    print('memory ', experiment, ' at time:', time_str)
    with open("runs/memory_" + experiment + "_compression_seq_mnist_" + time_str + ".txt", "w") as f:
        subprocess.call(['python', 'adding_memory_RNN_compression.py',
                         '--cell', 'WaveletGRU', '--compression_mode', experiment,
                         '--problem', 'memory'], stdout=f)

print('fastfood full compression')
time_str = str(datetime.datetime.today())
print('adding problem full compression', time_str)
with open("runs/fastfood_adding_problem_full_compression_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'adding',
                     '--cell', 'FastFoodGRU', '--compression_mode', 'full'], stdout=f)
time_str = str(datetime.datetime.today())
print('memory problem full compression', time_str)
with open("runs/fastfood_memory_problem_full_compression_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'memory',
                     '--cell', 'FastFoodGRU', '--compression_mode', 'full'], stdout=f)