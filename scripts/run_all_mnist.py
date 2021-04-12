# Created by moritz (wolter@cs.uni-bonn.de) at 17/12/2019
import datetime
import subprocess


experiment_total = 50
for exp in range(experiment_total):
    time_str = str(datetime.datetime.today())
    print('exp', exp, 'at', time_str)
    with open("runs/" + str(exp) + "_mnist_comp_" + time_str + ".txt", "w") as f:
        for drop in [0, 0.1, 0.2, 0.3]:
            subprocess.call(['python', 'mnist_compression.py',
                             '--compression', 'Wavelet', '--epochs', '20', '--gamma', '0.8',
                             '--wave_drop', str(drop)], stdout=f)
