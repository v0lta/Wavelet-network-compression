import numpy as np
import matplotlib.pyplot as plt

# construct a hermitian vector and look it it's fft.
r = np.array([1., 2.])
phi = np.pi*np.array([.25, 1.])
cr = np.concatenate([np.array([1.]), r, r[::-1]])
cphi = np.concatenate([np.array([0.]), phi, -phi[::-1]])
hermitian_symmetric_signal = cr*np.exp([-1j*cphi])[0, :]
freq_kernel = np.fft.fft(hermitian_symmetric_signal)
signal = np.random.uniform(0., 10., size=[5])
freq_signal = np.fft.fft(signal)
result = np.fft.ifft(freq_signal*freq_kernel)

# explore the half spectral weight formulation idea.
# complex weights -> fft -> add Hermitian half of spectrum. What does that mean?
length = 1000
t = np.linspace(0, 2*np.pi, length)
sine = np.sin(t)
zeroes = np.zeros([length])
input_signal = np.concatenate([zeroes, sine, zeroes, sine, zeroes])
to_pad = input_signal.shape[-1] - sine.shape[-1]
pad_kernel = np.pad(sine, (0, to_pad))

kernel_spectrum = np.fft.fft(pad_kernel)
signal_spectrum = np.fft.fft(input_signal)

conv = np.fft.ifft(kernel_spectrum*signal_spectrum)
plt.plot(input_signal)
plt.plot(pad_kernel)
plt.plot(conv/np.max(np.abs(conv)))
plt.show()