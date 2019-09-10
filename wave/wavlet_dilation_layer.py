'''
Wavelet dilation and sparse matrix multiplication.
'''
import torch


class Wave1D(torch.nn.Module):
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self):
        '''
        Compute the wavelet transform followed by sparse matrix multiplication.
        Perform an inverse transform.
        '''
        pass

