# Created by moritz (wolter@cs.uni-bonn.de) at 12/12/2019

import datetime
import subprocess

subprocess.call('pwd')
hidden = 64
epochs = 50
print(hidden, epochs)
# print('seq_mnist baseline GRU')
time_str = str(datetime.datetime.today())
print('time:', time_str)
with open("runs/v4/baselineGRU_seq_mnist_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                     '--hidden', str(hidden), '--epochs', str(epochs)],
                    stdout=f)

experiment_lst = ['state_reset', 'state_update', 'gates', 'full',
                  'update', 'state', 'reset']
print('seq_mnist wavelet compression')
for experiment in experiment_lst:
    time_str = str(datetime.datetime.today())
    print(experiment, ' at time:', time_str)
    with open("runs/v4/" + experiment + "_compression_seq_mnist_" + time_str + ".txt", "w") as f:
        subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                         '--cell', 'WaveletGRU', '--compression_mode', experiment,
                         '--hidden', str(hidden), '--epochs', str(epochs)], stdout=f)

print('fastfood full compression')
time_str = str(datetime.datetime.today())
with open("runs/v4/fastfood_full_compression_seq_mnist_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                     '--cell', 'FastFoodGRU', '--hidden', str(hidden),
                     '--epochs', str(epochs)], stdout=f)
