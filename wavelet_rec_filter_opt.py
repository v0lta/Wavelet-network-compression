import pywt
import numpy as np
import collections
import torch

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from generators.mackey_glass import MackeyGenerator
from wave.learn_wave import Wave1D
import matplotlib2tikz as tikz


CustomWavelet = collections.namedtuple('Wavelet', ['dec_lo', 'dec_hi',
                                                   'rec_lo', 'rec_hi', 'name'])


print(torch.cuda.get_device_name(), torch.cuda.is_available())

pd = {}
pd['iterations'] = 8000
pd['tmax'] = 256
pd['delta_t'] = 0.1
pd['pred_samples'] = 256
pd['window_size'] = 512
pd['lr'] = 0.004
pd['batch_size'] = 256
pd['dropout'] = 0.0
pd['channels'] = [30, 30, 30, 30, 30, 30]
pd['overlap'] = int(pd['window_size']*0.5)
pd['step_size'] = pd['window_size'] - pd['overlap']
pd['fft_freq_no'] = int(pd['window_size']//2 + 1)*2  # *2 for C
pd['window_fun'] = 'hamming'

generator = MackeyGenerator(batch_size=pd['batch_size'],
                            tmax=pd['tmax'],
                            delta_t=pd['delta_t'])

mackey_data_1 = torch.squeeze(generator())

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
    mackey_data_1.unsqueeze(1).unsqueeze(1).cpu())
print('alias cancellation loss:',
      wave1d_8.alias_cancellation_loss().detach().numpy(), ',',
      wavelet.name)
print('perfect reconstruction loss:',
      wave1d_8.perfect_reconstruction_loss().detach().numpy())

# reconstruct the input
my_rec = wave1d_8.reconstruction(wave1d_8_freq)
print('my_rec error', np.sum(np.abs(my_rec[0, 0, 0, :].detach().numpy()
                                    - mackey_data_1[0, :].cpu().numpy())))

plt.plot(my_rec[0, 0, 0, :].detach().numpy())
plt.plot(mackey_data_1[0, :].cpu().numpy())
plt.plot(np.abs(my_rec[0, 0, 0, :].detach().numpy() - mackey_data_1[0, :].cpu().numpy()))
plt.show()


# wavelet compression.
# zero the low coefficients.
def zero_by_scale(wavelet_coeffs, zero_at=5):
    '''
    Simply zero out entire scales.
    :param wavelet_coeffs: The list re
    :param zero_at:
    :return: A list where some coefficients are zeroed out.
    '''
    coefficients_low = []
    for no, c in enumerate(wavelet_coeffs):
        if no > len(wave1d_8_freq) - zero_at:
            coefficients_low.append(c)
        else:
            coefficients_low.append(0*c)
    return coefficients_low


def zero_by_magnitude(wavelet_coeffs, cutoff_mag=5e-1):
    sparse_coefficients = []
    for no, c_vec in enumerate(wavelet_coeffs):
        mask = (torch.abs(c_vec) > cutoff_mag).type(torch.float32)
        c_vec_sparse = c_vec*mask
        sparse_coefficients.append(c_vec_sparse)
    return sparse_coefficients


zero_by_method = zero_by_magnitude

c_low = zero_by_method(wave1d_8_freq)

rec_low = wave1d_8.reconstruction(c_low)
print('rec_low error', np.sum(np.abs(rec_low[0, 0, 0, :].detach().numpy()
                                     - mackey_data_1[0, :].cpu().numpy())))

plt.title('haar')
plt.plot(rec_low[0, 0, 0, :].detach().numpy())
plt.plot(mackey_data_1[0, :].cpu().numpy())
plt.plot(np.abs(rec_low[0, 0, 0, :].detach().numpy() - mackey_data_1[0, :].cpu().numpy()))
# savefig('haar')
tikz.save('haar.tex', standalone=True)
plt.show()

plt.title('haar coefficients')
plt.semilogy(np.abs(torch.cat(wave1d_8_freq, -1)[0, 0, 0, :].detach().numpy()))
plt.semilogy(np.abs(torch.cat(c_low, -1)[0, 0, 0, :].detach().numpy()), '.')
tikz.save('haar_coefficients.tex', standalone=True)
plt.show()

# optimize the basis:
wave1d_8.cuda()
steps = pd['iterations']
opt = torch.optim.Adagrad(wave1d_8.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

rec_loss_lst = []
for s in range(steps):
    opt.zero_grad()
    mackey_data = torch.squeeze(generator())
    wave1d_8_freq = wave1d_8.analysis(
        mackey_data.unsqueeze(1).unsqueeze(1))

    c_low = zero_by_method(wave1d_8_freq)

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
    if s % 250 == 0:
        print(s, loss.detach().cpu().numpy(),
              'mse', msel.detach().cpu().numpy(),
              'acl', acl.detach().cpu().numpy(),
              'prl', prl.detach().cpu().numpy())


wave1d_8_freq = wave1d_8.analysis(
    mackey_data_1.unsqueeze(1).unsqueeze(1))

c_low = zero_by_method(wave1d_8_freq)

rec_low = wave1d_8.reconstruction(c_low)
print('rec_low error', np.sum(np.abs(rec_low[0, 0, 0, :].detach().cpu().numpy()
                                     - mackey_data_1[0, :].cpu().numpy())))

plt.title('Optimized Haar')
plt.plot(rec_low[0, 0, 0, :].detach().cpu().numpy())
plt.plot(mackey_data_1[0, :].cpu().numpy())
plt.plot(np.abs(rec_low[0, 0, 0, :].detach().cpu().numpy() - mackey_data_1[0, :].cpu().numpy()))
# plt.savefig('optimized_haar')
tikz.save('optimized_haar.tex', standalone=True)
plt.show()

plt.title('Optimized haar coefficients')
plt.semilogy(np.abs(torch.cat(wave1d_8_freq, -1)[0, 0, 0, :].detach().cpu().numpy()))
plt.semilogy(np.abs(torch.cat(c_low, -1)[0, 0, 0, :].detach().cpu().numpy()), '.')
tikz.save('optimized_haar_coefficients.tex', standalone=True)
plt.show()

plt.semilogy(rec_loss_lst[10:])
plt.show()
