import pywt
import numpy as np
import torch

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from generators.mackey_glass import MackeyGenerator
from wave.transform2d import DWTForward, DWTInverse
from wave.learn_wave import Wave1D

bpd = {}
bpd['iterations'] = 8000
bpd['tmax'] = 256
bpd['delta_t'] = 0.1
bpd['pred_samples'] = 256
bpd['window_size'] = 512
bpd['lr'] = 0.004
bpd['batch_size'] = 4
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

wavelet = pywt.Wavelet('haar')
cA, cD = pywt.dwt(mackey_data_numpy[0, :], wavelet=wavelet)
plt.plot(mackey_data_numpy[0, :])
plt.show()
plt.plot(cA, cD)
plt.title('pywt')
plt.show()

# try out the multiresolution code.
c_10 = pywt.wavedec(mackey_data_numpy[0, :], wavelet=wavelet, level=5)
print([c.shape for c in c_10])

# compare to pytorch implementation.
pywave_forward = DWTForward(J=1, wave=wavelet)
pywave_inverse = DWTInverse(wave=wavelet)

cA_pyt, cD_pyt = pywave_forward(mackey_data.unsqueeze(1).unsqueeze(1).cpu())
cA_pyt_numpy = cA_pyt[0, 0, 0, :].detach().numpy()
cD_pyt_numpy = cD_pyt[0][0, 0, 1, 0, :].detach().numpy()

diff_A = np.linalg.norm(cA - cA_pyt_numpy)
diff_D = np.linalg.norm(cD - cD_pyt_numpy)
print(diff_A, diff_D)
plt.plot(cA_pyt_numpy, cD_pyt_numpy)
plt.title('torchwave')
plt.show()

reconstruction = pywave_inverse((cA_pyt, cD_pyt))

# test my own code 1d 1 level.
wave1d = Wave1D(wavelet.dec_lo, wavelet.dec_hi, wavelet.rec_lo, wavelet.rec_hi,
                scales=1)
low, high = wave1d.analysis(mackey_data.unsqueeze(1).unsqueeze(1).cpu())
plt.plot(high[0, 0, 0, :].detach().cpu().numpy(),
         low[0, 0, 0, :].detach().cpu().numpy())
plt.title('my wave')
plt.show()
print(np.linalg.norm(cA - low[0, 0, 0, :].detach().cpu().numpy()))
print(np.linalg.norm(cD - high[0, 0, 0, :].detach().cpu().numpy()))
print('done')

# try out the multilevel version.
wave1d_10 = Wave1D(wavelet.dec_lo, wavelet.dec_hi, wavelet.rec_lo, wavelet.rec_hi,
                   scales=5)
wave1d_10r = wave1d_10.analysis(mackey_data.unsqueeze(1).unsqueeze(1).cpu())

for no, cp in enumerate(wave1d_10r):
    cp = cp[0, 0, 0, :].detach().numpy()
    c = c_10[len(c_10) - no - 1]
    print(np.linalg.norm(cp - c), c.shape)

# wavelet compression.
