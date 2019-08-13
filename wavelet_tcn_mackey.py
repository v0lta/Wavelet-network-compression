import time
import pywt
import numpy as np
import copy
import torch
import scipy.signal as scisig

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from generators.mackey_glass import MackeyGenerator
from temporal_conv_net import TemporalConvNet
from fourier.short_time_fourier_pytorch import stft, istft
import ipdb

bpd = {}
bpd['iterations'] = 8000
bpd['tmax'] = 512
bpd['delta_t'] = 1.0
bpd['pred_samples'] = 256
bpd['window_size'] = 128
bpd['lr'] = 0.004
bpd['batch_size'] = 32
bpd['dropout'] = 0.0
bpd['channels'] = [30, 30, 30, 30, 30, 30]
bpd['overlap'] = int(bpd['window_size']*0.5)
bpd['step_size'] = bpd['window_size'] - bpd['overlap']
bpd['fft_freq_no'] = int(bpd['window_size']//2 + 1)*2  # *2 for C
bpd['window_fun'] = 'hamming'

generator = MackeyGenerator(batch_size=bpd['batch_size'],
                            tmax=bpd['tmax'],
                            delta_t=bpd['delta_t'])

mackey_data = torch.squeeze(generator())

# try out the pywavelets wavelet transfrom.
mackey_data_numpy = mackey_data.detach().cpu().numpy()

cA, cD = pywt.dwt(mackey_data_numpy, 'db2')
