import time
import pywt
import numpy as np
import copy
import torch
import scipy.signal as scisig

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from generators.mackey_glass import MackeyGenerator
from temporal_conv_net import TemporalConvNet
from wave.transform2d import DWTForward, DWTInverse
from wave.learn_wave import Wave1D
import ipdb

print(torch.cuda.get_device_name(), torch.cuda.is_available())

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

# wavelet = pywt.Wavelet('haar')
# wavelet = pywt.Wavelet('bior2.4')
wavelet = pywt.Wavelet('db4')

p_lo = np.convolve(wavelet.filter_bank[0], wavelet.filter_bank[2])
p_hi = np.convolve(wavelet.filter_bank[1], wavelet.filter_bank[3])
p_sum = p_lo + p_hi
print('pr lo', np.sum(p_lo))
print('pr hi', np.sum(p_hi))

cA, cD = pywt.dwt(mackey_data_numpy[0, :], wavelet=wavelet)
plt.plot(mackey_data_numpy[0, :])
plt.show()
plt.loglog(cA, cD, '.')
plt.title('pywt')
plt.show()

# try out the multiresolution code.
c_10 = pywt.wavedec(mackey_data_numpy[0, :],
                    wavelet=wavelet, level=8, mode='zero')
print([c.shape for c in c_10])

# compare to pytorch implementation.
pywave_forward = DWTForward(J=1, wave=wavelet)
pywave_inverse = DWTInverse(wave=wavelet)

cA_pyt, cD_pyt = pywave_forward(mackey_data.unsqueeze(1).unsqueeze(1).cpu())
cA_pyt_numpy = cA_pyt[0, 0, 0, :].detach().numpy()
cD_pyt_numpy = cD_pyt[0][0, 0, 1, 0, :].detach().numpy()

diff_A = np.linalg.norm(cA - cA_pyt_numpy)
diff_D = np.linalg.norm(cD - cD_pyt_numpy)
print('cA diff', diff_A,
      'cD diff', diff_D)
plt.loglog(cA_pyt_numpy, cD_pyt_numpy, '.')
plt.title('torchwave')
plt.show()

reconstruction = pywave_inverse((cA_pyt, cD_pyt))

# test my own code 1d 1 level.
wave1d = Wave1D(wavelet, scales=1)
low, high = wave1d.analysis(mackey_data.unsqueeze(1).unsqueeze(1).cpu())
plt.loglog(high[0, 0, 0, :].detach().cpu().numpy(),
           low[0, 0, 0, :].detach().cpu().numpy(), '.')
plt.title('my wave')
plt.show()
print('low diff', np.linalg.norm(cD - low[0, 0, 0, :].detach().cpu().numpy()))
print('high diff', np.linalg.norm(
    cA - high[0, 0, 0, :].detach().cpu().numpy()))
print('done')

# try out the multilevel version.
wave1d_10 = Wave1D(wavelet, scales=8)
wave1d_10_freq = wave1d_10.analysis(
    mackey_data.unsqueeze(1).unsqueeze(1).cpu())
print('alias cancellation loss:',
      wave1d_10.alias_cancellation_loss().detach().numpy(), ',',
      wavelet.name)
print('perfect reconstruction loss:',
      wave1d_10.perfect_reconstruction_loss().detach().numpy())

for no, cp in enumerate(wave1d_10_freq):
    cp = cp[0, 0, 0, :].detach().numpy()
    c = c_10[len(c_10) - no - 1]
    print(np.linalg.norm(cp - c), c.shape)


# reconstruct the input
my_rec = wave1d_10.reconstruction(wave1d_10_freq)
print('my_rec error', np.sum(np.abs(my_rec[0, 0, 0, :].detach().numpy()
                                    - mackey_data[0, :].cpu().numpy())))

plt.plot(my_rec[0, 0, 0, :].detach().numpy())
plt.plot(mackey_data[0, :].cpu().numpy())
plt.plot(np.abs(my_rec[0, 0, 0, :].detach().numpy() - mackey_data[0, :].cpu().numpy()))
plt.show()


# wavelet compression.
# zero the low coefficients.
c_low = []
for no, c in enumerate(wave1d_10_freq):
    if no > len(wave1d_10_freq) - 5:
        c_low.append(c)
    else:
        c_low.append(0*c)

rec_low = wave1d_10.reconstruction(c_low)
print('rec_low error', np.sum(np.abs(rec_low[0, 0, 0, :].detach().numpy()
                                     - mackey_data[0, :].cpu().numpy())))

plt.plot(rec_low[0, 0, 0, :].detach().numpy())
plt.plot(mackey_data[0, :].cpu().numpy())
plt.plot(np.abs(rec_low[0, 0, 0, :].detach().numpy() - mackey_data[0, :].cpu().numpy()))
plt.show()

plt.semilogy(np.abs(torch.cat(wave1d_10_freq, -1)[0, 0, 0, :].detach().numpy()))
plt.semilogy(np.abs(torch.cat(c_low, -1)[0, 0, 0, :].detach().numpy()))
plt.show()
