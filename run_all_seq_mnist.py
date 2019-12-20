# Created by moritz (wolter@cs.uni-bonn.de) at 12/12/2019

import datetime
import subprocess
subprocess.call('pwd')
hidden = 256
print('seq_mnist baseline GRU')
time_str = str(datetime.datetime.today())
print('time:', time_str)
with open("runs/baselineGRU_seq_mnist_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py', '--hidden', str(hidden)],
                    stdout=f)

experiment_lst = ['state', 'update', 'reset', 'state_reset', 'state_update', 'gates', 'full']
print('seq_mnist wavelet compression')
for experiment in experiment_lst:
    time_str = str(datetime.datetime.today())
    print(experiment, ' at time:', time_str)
    with open("runs/" + experiment + "_compression_seq_mnist_" + time_str + ".txt", "w") as f:
        subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                         '--cell', 'WaveletGRU', '--compression_mode', experiment,
                         '--hidden', str(hidden)], stdout=f)

print('fastfood full compression')
with open("runs/fastfood_full_compression_seq_mnist.txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                     '--cell', 'FastFoodGRU', '--compression_mode', 'full',
                     '--hidden', str(hidden)], stdout=f)
