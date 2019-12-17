# Created by moritz (wolter@cs.uni-bonn.de) at 17/12/2019
import datetime
import subprocess


experiment_total = 100
for exp in range(experiment_total):
    time_str = str(datetime.datetime.today())
    print('exp', exp, 'at', time_str)
    with open("runs/" + str(exp) + "_mnist_comp_" + time_str + ".txt", "w") as f:
        subprocess.call(['python', 'mnist_compression.py',
                         '--compression', 'Wavelet', '--epochs', '100', '--gamma', '0.9'], stdout=f)
