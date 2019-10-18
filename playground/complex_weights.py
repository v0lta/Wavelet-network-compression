import time
import numpy as np
import scipy.fftpack as scifft
import matplotlib.pyplot as plt
import torch

# construct a hermitian vector and look it it's fft.
# r = np.array([1., 2.])
# phi = np.pi*np.array([.25, 1.])
# cr = np.concatenate([np.array([1.]), r, r[::-1]])
# cphi = np.concatenate([np.array([0.]), phi, -phi[::-1]])
# hermitian_symmetric_signal = cr*np.exp([-1j*cphi])[0, :]
# freq_kernel = np.fft.fft(hermitian_symmetric_signal)
# signal = np.random.uniform(0., 10., size=[5])
# freq_signal = np.fft.fft(signal)
# result = np.fft.ifft(freq_signal*freq_kernel)
#
# # explore the half spectral weight formulation idea.
# # complex weights -> fft -> add Hermitian half of spectrum. What does that mean?
# length = 1000
# t = np.linspace(0, 2*np.pi, length)
# sine = np.sin(t)
# zeroes = np.zeros([length])
# input_signal = np.concatenate([zeroes, sine, zeroes, sine, zeroes])
# to_pad = input_signal.shape[-1] - sine.shape[-1]
# pad_kernel = np.pad(sine, (0, to_pad))
# kernel_spectrum = np.fft.fft(pad_kernel)
# signal_spectrum = np.fft.fft(input_signal)
# conv = np.fft.ifft(kernel_spectrum*signal_spectrum)
# plt.plot(input_signal)
# plt.plot(pad_kernel)
# plt.plot(conv/np.max(np.abs(conv)))
# plt.show()


# kernel = [4, 5, 6, 1]
# spec_kernel = np.fft.rfft(kernel)

# spec_weights = [14, -1+4j, 3+0j]
# time_weights = np.fft.irfft(spec_weights)

time_fft = []
time_norm = []
for size in range(100000):
    rand = np.random.uniform(size=[100000])
    start_norm = time.time()
    norm = np.linalg.norm(rand)
    stop_norm = time.time() - start_norm
    start_fft = time.time()
    fft = scifft.fft(rand)
    stop_fft = time.time() - start_fft
    time_fft.append(stop_fft)
    time_norm.append(stop_norm)
    if size % 5000 == 0:
        print(size, 'norm', stop_norm, 'fft', stop_fft)

plt.plot(time_norm)
plt.plot(time_fft)
plt.show()

# TODO: Try on GPU in pytorch
# time_fft = []
# time_norm = []
#
# for size in range(100000):
#     rand = torch.tensor(shape=[size])
#     rand = torch.random.uniform_(rand, )

