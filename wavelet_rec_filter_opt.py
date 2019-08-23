import time
import pywt
import numpy as np
import collections
import torch
import scipy.signal as scisig

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from generators.mackey_glass import MackeyGenerator
from temporal_conv_net import TemporalConvNet
from wave.transform2d import DWTForward, DWTInverse
from wave.learn_wave import Wave1D


CustomWavelet = collections.namedtuple('Wavelet', ['dec_lo', 'dec_hi',
                                                   'rec_lo', 'rec_hi', 'name'])


print(torch.cuda.get_device_name(), torch.cuda.is_available())

bpd = {}
bpd['iterations'] = 8000
bpd['tmax'] = 256
bpd['delta_t'] = 0.1
bpd['pred_samples'] = 256
bpd['window_size'] = 512
bpd['lr'] = 0.004
bpd['batch_size'] = 256
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

wavelet = pywt.Wavelet('haar')
# wavelet = pywt.Wavelet('db4')
# wavelet = pywt.Wavelet('bior2.4')

# TODO: Start from noise here! Or perhaps a haar wavelet which turns into something else.
wavelet = CustomWavelet(dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                        dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
                        rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                        rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
                        name='custom')


# try out the multilevel version.
wave1d_8 = Wave1D(wavelet, scales=8)
wave1d_8_freq = wave1d_8.analysis(
    mackey_data.unsqueeze(1).unsqueeze(1).cpu())
print('alias cancellation loss:',
      wave1d_8.alias_cancellation_loss().detach().numpy(), ',',
      wavelet.name)
print('perfect reconstruction loss:',
      wave1d_8.perfect_reconstruction_loss().detach().numpy())

# reconstruct the input
my_rec = wave1d_8.reconstruction(wave1d_8_freq)
print('my_rec error', np.sum(np.abs(my_rec[0, 0, 0, :].detach().numpy()
                                    - mackey_data[0, :].cpu().numpy())))

plt.plot(my_rec[0, 0, 0, :].detach().numpy())
plt.plot(mackey_data[0, :].cpu().numpy())
plt.plot(np.abs(my_rec[0, 0, 0, :].detach().numpy() - mackey_data[0, :].cpu().numpy()))
plt.show()


# wavelet compression.
# zero the low coefficients.
c_low = []
for no, c in enumerate(wave1d_8_freq):
    if no > len(wave1d_8_freq) - 5:
        c_low.append(c)
    else:
        c_low.append(0*c)

rec_low = wave1d_8.reconstruction(c_low)
print('rec_low error', np.sum(np.abs(rec_low[0, 0, 0, :].detach().numpy()
                                     - mackey_data[0, :].cpu().numpy())))

plt.plot(rec_low[0, 0, 0, :].detach().numpy())
plt.plot(mackey_data[0, :].cpu().numpy())
plt.plot(np.abs(rec_low[0, 0, 0, :].detach().numpy() - mackey_data[0, :].cpu().numpy()))
plt.show()

plt.semilogy(np.abs(torch.cat(wave1d_8_freq, -1)[0, 0, 0, :].detach().numpy()))
plt.semilogy(np.abs(torch.cat(c_low, -1)[0, 0, 0, :].detach().numpy()))
plt.show()

# optimize the basis:
wave1d_8.cuda()
steps = 600
opt = torch.optim.Adagrad(wave1d_8.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

rec_loss_lst = []
for s in range(steps):
    opt.zero_grad()
    mackey_data = torch.squeeze(generator())
    wave1d_8_freq = wave1d_8.analysis(
        mackey_data.unsqueeze(1).unsqueeze(1))

    c_low = []
    for no, c in enumerate(wave1d_8_freq):
        if no > len(wave1d_8_freq) - 6:
            c_low.append(c)
        else:
            c_low.append(0*c)

    rec_low = wave1d_8.reconstruction(c_low)
    msel = criterion(mackey_data, torch.squeeze(rec_low))
    loss = msel
    acl = wave1d_8.alias_cancellation_loss()
    prl = wave1d_8.perfect_reconstruction_loss()
    loss += (acl + prl) # * s/steps

    # compute gradients
    loss.backward()
    # apply gradients
    opt.step()
    rec_loss_lst.append(msel.detach().cpu().numpy())
    print(s, loss.detach().cpu().numpy(),
          'mse', msel.detach().cpu().numpy(),
          'acl', acl.detach().cpu().numpy(),
          'prl', prl.detach().cpu().numpy())

rec_low = wave1d_8.reconstruction(c_low)
print('rec_low error', np.sum(np.abs(rec_low[0, 0, 0, :].detach().cpu().numpy()
                                     - mackey_data[0, :].cpu().numpy())))

plt.plot(rec_low[0, 0, 0, :].detach().cpu().numpy())
plt.plot(mackey_data[0, :].cpu().numpy())
plt.plot(np.abs(rec_low[0, 0, 0, :].detach().cpu().numpy() - mackey_data[0, :].cpu().numpy()))
plt.show()
plt.semilogy(rec_loss_lst)
plt.show()