# Created by moritz (wolter@cs.uni-bonn.de) at 12/12/2019

import subprocess
subprocess.call('pwd')

print('baseline GRU')
with open("runs/baselineGRU_seq_mnist_seq_mnist.txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py'], stdout=f)
print('state compression')
with open("runs/state_compression_seq_mnist.txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                     '--cell', 'WaveletGRU', '--compression_mode', 'state'], stdout=f)
print('update compression')
with open("runs/update_compression_seq_mnist.txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                     '--cell', 'WaveletGRU', '--compression_mode', 'update'], stdout=f)
print('reset compression')
with open("runs/reset_compression_seq_mnist.txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                     '--cell', 'WaveletGRU', '--compression_mode', 'reset'], stdout=f)
print('state_reset compression')
with open("runs/state_reset_compression_seq_mnist.txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                     '--cell', 'WaveletGRU', '--compression_mode', 'state_reset'], stdout=f)
print('state_update compression')
with open("runs/state_update_compression_seq_mnist.txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                     '--cell', 'WaveletGRU', '--compression_mode', 'state_update'], stdout=f)
print('gates compression')
with open("runs/gates_compression_seq_mnist.txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                     '--cell', 'WaveletGRU', '--compression_mode', 'gates'], stdout=f)
print('full compression')
with open("runs/full_compression_seq_mnist.txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py',
                     '--cell', 'WaveletGRU', '--compression_mode', 'full'], stdout=f)
print('fastfood full compression')
with open("runs/fastfood_full_compression_seq_mnist.txt", "w") as f:
    subprocess.call(['python', 'sequential_cifar_mnist_compression.py', '--problem', 'adding',
                     '--cell', 'FastFoodGRU', '--compression_mode', 'full'], stdout=f)
