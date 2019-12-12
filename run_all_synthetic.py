# Created by moritz (wolter@cs.uni-bonn.de) at 11/12/2019

import subprocess
subprocess.call('pwd')

print('baseline GRU')
print('adding problem baseline')
with open("runs/baselineGRU_adding.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'adding'], stdout=f)
print('memory problem baseline')
with open("runs/baselineGRU_memory.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'memory'], stdout=f)

print('state compression')
print('adding problem state compression')
with open("runs/adding_problem_state_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'adding',
                     '--cell', 'WaveletGRU', '--compression_mode', 'state'], stdout=f)
print('memory problem state compression')
with open("runs/memory_problem_state_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'memory',
                     '--cell', 'WaveletGRU', '--compression_mode', 'state'], stdout=f)

print('update compression')
print('adding problem update compression')
with open("runs/adding_problem_update_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'adding',
                     '--cell', 'WaveletGRU', '--compression_mode', 'update'], stdout=f)
print('memory problem update compression')
with open("runs/memory_problem_update_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'memory',
                     '--cell', 'WaveletGRU', '--compression_mode', 'update'], stdout=f)

print('reset compression')
print('adding problem reset compression')
with open("runs/adding_problem_reset_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'adding',
                     '--cell', 'WaveletGRU', '--compression_mode', 'reset'], stdout=f)
print('memory problem reset compression')
with open("runs/memory_problem_reset_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'memory',
                     '--cell', 'WaveletGRU', '--compression_mode', 'reset'], stdout=f)

print('state_reset compression')
print('adding problem state_reset compression')
with open("runs/adding_problem_state_reset_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'adding',
                     '--cell', 'WaveletGRU', '--compression_mode', 'state_reset'], stdout=f)
print('memory problem state_reset compression')
with open("runs/memory_problem_state_reset_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'memory',
                     '--cell', 'WaveletGRU', '--compression_mode', 'state_reset'], stdout=f)

print('state_update compression')
print('adding problem state_update compression')
with open("runs/adding_problem_state_update_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'adding',
                     '--cell', 'WaveletGRU', '--compression_mode', 'state_update'], stdout=f)
print('memory problem state_update compression')
with open("runs/memory_problem_state_update_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'memory',
                     '--cell', 'WaveletGRU', '--compression_mode', 'state_update'], stdout=f)


print('gates compression')
print('adding problem gates compression')
with open("runs/adding_problem_gates_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'adding',
                     '--cell', 'WaveletGRU', '--compression_mode', 'gates'], stdout=f)
print('memory problem gates compression')
with open("runs/memory_problem_gates_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'memory',
                     '--cell', 'WaveletGRU', '--compression_mode', 'gates'], stdout=f)

print('full compression')
print('adding problem full compression')
with open("runs/adding_problem_full_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'adding',
                     '--cell', 'WaveletGRU', '--compression_mode', 'full'], stdout=f)
print('memory problem full compression')
with open("runs/memory_problem_full_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'memory',
                     '--cell', 'WaveletGRU', '--compression_mode', 'full'], stdout=f)

print('fastfood full compression')
print('adding problem full compression')
with open("runs/fastfood_adding_problem_full_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'adding',
                     '--cell', 'FastFoodGRU', '--compression_mode', 'full'], stdout=f)
print('memory problem full compression')
with open("runs/fastfood_memory_problem_full_compression.txt", "w") as f:
    subprocess.call(['python', 'adding_memory_RNN_compression.py', '--problem', 'memory',
                     '--cell', 'FastFoodGRU', '--compression_mode', 'full'], stdout=f)