# implement an inverse STFT for pytorch.
import torch
import numpy as np
import scipy.signal as scisig
import warnings
# import ipdb


def stft_from_torch(input_tensor, nperseg, nstep=None, win_length=None,
                    window=None, center=True, pad_mode='reflect',
                    normalized=False, onesided=True):
    '''
    A torch stft implementation capable of handling multidimensional
    input tensors, with the last dimension being the time dimension.
    Warning:
        I don't get the scaling that's applied in torch.stft
        and therefore could not invert it properly.
    '''
    input_tensor_shape = input_tensor.shape
    # TODO put in condition on ndim > 2
    input_tensor = input_tensor.reshape([-1, input_tensor_shape[-1]])
    # torch.stft allows only inputs of shape [stack, time]
    Zxx = torch.stft(input_tensor, nperseg, nstep, win_length, window,
                     center, pad_mode, normalized, onesided)
    # restore the leading dimensions
    Zxx_shape = Zxx.shape
    Zxx = Zxx.reshape(input_tensor_shape[:-1] + Zxx_shape[-3:])
    return Zxx


def zero_ext(x, n, dim=-1):
    """
    Following:
    https://github.com/scipy/scipy/blob/master/scipy/signal/_arraytools.py
    Zero padding at the boundaries of an array
    Generate a new tensor that is a zero padded extension of `x` along
    a speficied dimension.
    """
    if n < 1:
        return x
    zeros_shape = list(x.shape)
    zeros_shape[dim] = n
    zeros = torch.zeros(zeros_shape, dtype=x.dtype, device=x.device)
    ext = torch.cat((zeros, x, zeros), dim=dim)
    return ext


def stft(x, window, nperseg=None, noverlap=None,
         boundary=None, padded=True):
    '''
    Imlement what's happening in scipy's stft implementation.

    Returns:
        freq_x: [batch_size, freqs, time, 2]
    '''
    assert type(window) == torch.Tensor
    assert len(window.shape) == 1

    boundary_funcs = {'zeros': zero_ext,
                      None: None}

    if boundary not in boundary_funcs:
        raise ValueError("Unknown boundary option '{0}', must be one of: {1}"
                         .format(boundary, list(boundary_funcs.keys())))

    if nperseg is None:
        nperseg = len(window)
    else:
        assert len(window) == nperseg

    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg // 2, dim=-1)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        nadd = (-(x.shape[-1] - nperseg) % nstep) % nperseg
        if nadd > 0:
            zeros_shape = list(x.shape[:-1]) + [nadd]
            x = torch.cat((x, torch.zeros(zeros_shape, dtype=x.dtype,
                                          device=x.device)), dim=-1)

    scale = 1.0 / (torch.sum(window)*torch.sum(window))

    scale = torch.sqrt(scale)

    x_framed = x.unfold(-1, nperseg, nstep)
    x_framed = x_framed*window
    result = torch.rfft(x_framed, signal_ndim=1, normalized=False,
                        onesided=True)
    result = result*scale
    result = result.transpose(-2, -3)
    return result


def istft(Zxx, window, nperseg=None, noverlap=None,
          boundary=True, epsilon=None):
    '''
        1-Dimensional multi-channel inverse Short time Fourier transform.
        Inspired by:
        https://github.com/scipy/scipy/blob/v1.3.0/scipy/signal/spectral.py
    Params:
        Zxx: Frequency domain data the output of torch.STFT
            -> [batch, freq, time, 2].
        window: A torch tensor containing the window i.e from torch.hann_window
        nperseg: The lenght of the time-segments, should be equal to the window
                 length.
        nstep: The step size of the STFT, also referred to as hop_length.
        boundary: If the bool is True padding equal to window_size/2 is
                  removed from the edges of the window.
        epsilon: If not None this number is added to the window, which can
                 improve numercial stability i.e. if a learned window function
                 is used.
    Returns:
        The original signal x.
    '''
    Zxx = Zxx.transpose(-3, -2)
    freq_axis = -2
    time_axis = -3

    nseg = Zxx.shape[time_axis]

    # Assume a onesided input and even segment length
    n_default = 2 * (Zxx.shape[freq_axis] - 1)

    # Check windowing parameters
    if nperseg is None:
        nperseg = n_default
    else:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')

    xsubs = torch.irfft(Zxx, signal_ndim=1, onesided=True,
                        normalized=False, signal_sizes=[nperseg])

    xsubs *= torch.sum(window)  # This takes care of the 'spectrum' scaling
    xsubs = xsubs.transpose(-1, -2)

    # Initialize output and normalization arrays
    outputlength = nperseg + (nseg-1)*nstep
    x = torch.zeros(list(Zxx.shape[:-3]) + [outputlength], dtype=xsubs.dtype,
                    device=xsubs.device)
    norm = torch.zeros(outputlength, dtype=xsubs.dtype, device=xsubs.device)

    # Construct the output from the ifft segments
    # This loop could perhaps be vectorized/strided somehow...
    for ii in range(nseg):
        # Window the ifft
        x[..., ii*nstep:ii*nstep+nperseg] += xsubs[..., ii] * window
        norm[..., ii*nstep:ii*nstep+nperseg] += torch.pow(window, 2)

    # Remove extension points
    if boundary:
        x = x[..., nperseg//2:-(nperseg//2)]
        norm = norm[..., nperseg//2:-(nperseg//2)]

    if torch.sum(norm > 1e-10) != len(norm):
        warnings.warn("NOLA condition failed, STFT may not be invertible")

    if epsilon:
        x /= (norm + epsilon)
    else:
        x /= torch.where(norm.gt(1e-10), norm, torch.ones_like(norm))

    return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def generate_data(tmax=20, delta_t=0.01, sigma=10.0,
                      beta=8.0/3.0, rho=28.0, batch_size=100,
                      rnd=True, dtype=torch.float32):
        """
        Generate synthetic training data using the Lorenz system
        of equations (https://en.wikipedia.org/wiki/Lorenz_system):
        dxdt = sigma*(y - x)
        dydt = x * (rho - z) - y
        dzdt = x*y - beta*z
        The system is simulated using a forward euler scheme
        (https://en.wikipedia.org/wiki/Euler_method).
        Params:
            tmax: The simulation time.
            delta_t: The step size.
            sigma: The first Lorenz parameter.
            beta: The second Lorenz parameter.
            rho: The thirs Lorenz parameter.
            batch_size: The first batch dimension.
            rnd: If true the lorenz seed is random.
        Returns:
            spikes: A Tensor of shape [batch_size, time, 1],
            states: A Tensor of shape [batch_size, time, 3].
        """
        # multi-dimensional data.
        def lorenz(x, t):
            return torch.stack([sigma*(x[:, 1] - x[:, 0]),
                                x[:, 0]*(rho - x[:, 2]) - x[:, 1],
                                x[:, 0]*x[:, 1] - beta*x[:, 2]],
                               dim=1)

        state0 = np.array([8.0, 6.0, 30.0])
        state0 = np.stack(batch_size*[state0], axis=0)
        if rnd:
            print('Lorenz initial state is random.')
            # ipdb.set_trace()
            state0 += np.random.uniform(-4, 4, [batch_size, 3])
        else:
            add_lst = []
            for i in range(batch_size):
                add_lst.append([0, float(i)*(1.0/batch_size), 0])
            add_tensor = np.stack(add_lst, axis=0)
            state0 += add_tensor
        states = [torch.from_numpy(state0)]

        for _ in range(int(tmax/delta_t)):
            states.append(states[-1] + delta_t*lorenz(states[-1], None))
        states = torch.stack(states, dim=1)

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot(states[:, 0], states[:, 1], states[:, 2],
        #         label='lorenz curve')
        # plt.show()

        # single dimensional data.
        spikes = torch.unsqueeze(torch.pow(states[:, :, 0], 2), -1)
        # normalize
        states = states/torch.max(torch.abs(states).flatten())
        spikes = spikes/torch.max(torch.abs(spikes).flatten())
        return spikes.type(dtype), states.type(dtype)

    def complex_abs(complex_tensor):
        r = complex_tensor[..., 0]
        c = complex_tensor[..., 1]
        return torch.sqrt(r*r + c*c)

    pd = {}
    pd['tmax'] = 10.23
    pd['delta_t'] = 0.01
    pd['batch_size'] = 100
    pd['input_samples'] = int(pd['tmax']/pd['delta_t'])+1
    spikes, states = generate_data(pd['tmax'], pd['delta_t'],
                                   batch_size=pd['batch_size'],
                                   rnd=False, dtype=torch.float64)
    print(spikes.shape)
    print(spikes[0, 100])
    pd['window_size'] = 128
    spikes_T = torch.squeeze(spikes)
    # spikes_T = spikes_T.permute(1, 0)
    window = torch.from_numpy(scisig.get_window('hann', Nx=pd['window_size']))
    window = window.type(torch.float64)

    # spikes_freq = torch.stft(spikes_T, window=window,
    #                            n_fft=pd['window_size'],
    #                          hop_length=pd['window_size']//2)
    spikes_freq = stft(spikes_T, window, nperseg=pd['window_size'],
                       noverlap=64, boundary='zeros', padded=False)

    # compare to scipy
    scipy_freq = scisig.stft(spikes_T.detach().cpu().numpy(),
                             window=window.detach().cpu().numpy(),
                             nperseg=pd['window_size'],
                             noverlap=64, boundary='zeros', padded=False)[-1]

    numpy_spikes_freq = spikes_freq.detach().cpu().numpy()
    numpy_spikes_freq = numpy_spikes_freq[..., 0] \
        + 1j*numpy_spikes_freq[..., 1]
    freq_loss = np.mean(np.abs(scipy_freq - numpy_spikes_freq))
    print('freq_loss 1d', freq_loss)  # We are doing ok here! :-D.

    spikes_abs = complex_abs(spikes_freq)
    # plt.imshow(spikes_abs[0, :, :].detach().cpu().numpy())
    # plt.show()
    # run the istft

    reconstruction = istft(spikes_freq, window=window,
                           nperseg=128, noverlap=64,
                           boundary=True)

    scipy_reconstruction = scisig.istft(scipy_freq, window=window.detach().cpu().numpy(),
                                        nperseg=128, noverlap=64,
                                        boundary=True)[-1]

    spikes = spikes.detach().cpu().numpy()
    spikes_rec = reconstruction.detach().cpu().numpy()
    # plt.plot(spikes[0, :, 0])
    # plt.plot(spikes_rec[0, :])
    # plt.show()
    print('error 1d', np.mean(spikes - np.expand_dims(spikes_rec, -1)))
    print('error 1d scipy',
          np.mean(spikes - np.expand_dims(scipy_reconstruction, -1)))

    # test the 3d case.
    # move time to the last dimension.
    states = states.permute([0, 2, 1])
    states_freq = stft(states, window=window,
                       nperseg=pd['window_size'],
                       noverlap=64, boundary='zeros',
                       padded=False)
    scipy_states_freq = scisig.stft(states.detach().cpu().numpy(),
                                    window=window.detach().cpu().numpy(),
                                    nperseg=pd['window_size'],
                                    noverlap=64, boundary='zeros', padded=False)[-1]
    numpy_states_freq = states_freq.detach().cpu().numpy()
    numpy_states_freq = numpy_states_freq[..., 0] \
        + 1j*numpy_states_freq[..., 1]
    freq_loss = np.mean(np.abs(numpy_states_freq - scipy_states_freq))
    print('freq_loss 3d', freq_loss)  # We are doing ok here! :-D.

    states_rec = istft(states_freq, window=window)

    from mpl_toolkits.mplot3d import Axes3D
    states = states.detach().cpu().numpy()
    states_rec = states_rec.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[0, 0, :], states[0, 1, :], states[0, 2, :])
    ax.plot(states_rec[0, 0, :], states_rec[0, 1, :], states_rec[0, 2, :])
    plt.show()
    print('error 3d', np.mean(states - states_rec))

    # ---- test the padding option. ----

    pd['tmax'] = 9.99
    pd['delta_t'] = 0.01
    pd['batch_size'] = 100
    pd['input_samples'] = int(pd['tmax']/pd['delta_t'])+1
    spikes, states = generate_data(pd['tmax'], pd['delta_t'],
                                   batch_size=pd['batch_size'],
                                   rnd=False, dtype=torch.float64)
    spikes_T = torch.squeeze(spikes)
    spikes_freq = stft(spikes_T, window, nperseg=pd['window_size'],
                       noverlap=64, boundary='zeros', padded=True)
    # compare to scipy
    scipy_freq = scisig.stft(spikes_T.detach().cpu().numpy(),
                             window=window.detach().cpu().numpy(),
                             nperseg=pd['window_size'],
                             noverlap=64, boundary='zeros', padded=True)[-1]

    numpy_spikes_freq = spikes_freq.detach().cpu().numpy()
    numpy_spikes_freq = numpy_spikes_freq[..., 0] \
        + 1j*numpy_spikes_freq[..., 1]
    freq_loss = np.mean(np.abs(scipy_freq - numpy_spikes_freq))
    print('freq_loss 1d, pad', freq_loss)  # We are doing ok here! :-D.

    spikes_abs = complex_abs(spikes_freq)
    # plt.imshow(spikes_abs[0, :, :].detach().cpu().numpy())
    # plt.show()
    # run the istft

    reconstruction = istft(spikes_freq, window=window,
                           nperseg=128, noverlap=64,
                           boundary=True)

    scipy_reconstruction = scisig.istft(scipy_freq, window=window.detach().cpu().numpy(),
                                        nperseg=128, noverlap=64,
                                        boundary=True)[-1]

    spikes = spikes.detach().cpu().numpy()
    spikes_rec = reconstruction.detach().cpu().numpy()
    # plt.plot(spikes[0, :, 0])
    # plt.plot(spikes_rec[0, :])
    # plt.show()
    print('error 1d, pad', np.mean(spikes - np.expand_dims(spikes_rec[:, :1000], -1)))
    print('error 1d scipy, pad',
          np.mean(spikes - np.expand_dims(scipy_reconstruction[:, :1000], -1)))
