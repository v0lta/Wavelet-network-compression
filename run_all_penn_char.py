# Created by moritz (wolter@cs.uni-bonn.de) at 12/12/2019
import datetime

import subprocess
subprocess.call('pwd')

print('baseline GRU')
print('penn char baseline')
time_str = str(datetime.datetime.today())
print('time:', time_str)
with open("runs/baselineGRU_penn_char_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'penn_test.py'], stdout=f)
print('state compression')
time_str = str(datetime.datetime.today())
print('time:', time_str)
with open("runs/state_compression_penn_char_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'penn_test.py',
                     '--cell', 'WaveGRU', '--compression_mode', 'state'], stdout=f)
print('update compression')
print('update compression')
time_str = str(datetime.datetime.today())
print('time:', time_str)
with open("runs/update_compression_penn_char_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'penn_test.py',
                     '--cell', 'WaveGRU', '--compression_mode', 'update'], stdout=f)
print('reset compression')
time_str = str(datetime.datetime.today())
print('time:', time_str)
with open("runs/reset_compression_penn_char_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'penn_test.py',
                     '--cell', 'WaveGRU', '--compression_mode', 'reset'], stdout=f)
print('state_reset compression')
time_str = str(datetime.datetime.today())
print('time:', time_str)
with open("runs/state_reset_compression_penn_char_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'penn_test.py',
                     '--cell', 'WaveGRU', '--compression_mode', 'state_reset'], stdout=f)
print('state_update compression')
time_str = str(datetime.datetime.today())
print('time:', time_str)
with open("runs/state_update_compression_penn_char_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'penn_test.py',
                     '--cell', 'WaveGRU', '--compression_mode', 'state_update'], stdout=f)
print('gates compression')
time_str = str(datetime.datetime.today())
print('time:', time_str)
with open("runs/gates_compression_penn_char_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'penn_test.py',
                     '--cell', 'WaveGRU', '--compression_mode', 'gates'], stdout=f)
print('full compression')
time_str = str(datetime.datetime.today())
print('time:', time_str)
with open("runs/full_compression_penn_char_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'penn_test.py',
                     '--cell', 'WaveGRU', '--compression_mode', 'full'], stdout=f)
print('fastfood full compression')
time_str = str(datetime.datetime.today())
print('time:', time_str)
with open("runs/fastfood_full_compression_penn_char_" + time_str + ".txt", "w") as f:
    subprocess.call(['python', 'penn_test.py', '--problem', 'adding',
                     '--cell', 'FastFoodGRU', '--compression_mode', 'full'], stdout=f)
