import datetime
import subprocess

experiment_lst = ['state_reset', 'state_reset', 'state_reset']
print('seq_mnist wavelet compression')
for experiment in experiment_lst:
    time_str = str(datetime.datetime.today())
    print(experiment, ' at time:', time_str)
    with open("runs/v4_" + experiment + "_compression_seq_mnist_" + time_str + ".txt", "w") as f:
        subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                         '--cell', 'WaveletGRU', '--compression_mode', experiment,
                         '--hidden', str(58), '--epochs', str(50)], stdout=f)
