# Created by moritz (wolter@cs.uni-bonn.de) at 12/12/2019
import datetime

import subprocess
subprocess.call('pwd')

# print('baseline GRU')
# print('penn char baseline')
# time_str = str(datetime.datetime.today())
# print('time:', time_str)
# with open("runs/baselineGRU_penn_char_" + time_str + ".txt", "w") as f:
#     subprocess.call(['python', 'penn_test.py'], stdout=f)

experiment_lst = ['state', 'reset', 'state_reset', 'state_update', 'full',
                  'gates', 'update']
print('wavelet compression')
for experiment in experiment_lst:
    time_str = str(datetime.datetime.today())
    print(experiment, ' at time:', time_str)
    with open("runs/v3_" + experiment + "_compression_penn_char_" + time_str + ".txt", "w") as f:
        subprocess.call(['python', 'penn_test.py',
                         '--cell', 'WaveletGRU', '--compression_mode', experiment], stdout=f)

# print('fastfood full compression')
# time_str = str(datetime.datetime.today())
# print('time:', time_str)
# with open("runs/v3_fastfood_full_compression_penn_char_" + time_str + ".txt", "w") as f:
#     subprocess.call(['python', 'penn_test.py', '--problem', 'adding',
#                      '--cell', 'FastFoodGRU', '--compression_mode', 'full'], stdout=f)
