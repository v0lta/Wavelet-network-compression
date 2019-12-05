# Created by moritz (wolter@cs.uni-bonn.de) at 05/12/2019
import torch
import collections
from RNN_compression.cells import GRUCell, FastFoodGRU, WaveletGRU
from timit_exp.timit_reader import TIMITDataSet

CustomWavelet = collections.namedtuple('Wavelet', ['dec_lo', 'dec_hi',
                                                   'rec_lo', 'rec_hi', 'name'])

pd = {}
pd['epochs'] = 2
pd['cell'] = 'GRU'
pd['lr'] = 0.001
pd['hidden'] = 0

timit = TIMITDataSet()
input_size = 13
output_size = 51

if pd['cell'] == 'WaveletGRU':
    # pd['init_wavelet'] = pywt.Wavelet('db6')
    pd['init_wavelet'] = CustomWavelet(dec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                       dec_hi=[0, 0, -0.7071067811865476, 0.7071067811865476, 0, 0],
                                       rec_lo=[0, 0, 0.7071067811865476, 0.7071067811865476, 0, 0],
                                       rec_hi=[0, 0, 0.7071067811865476, -0.7071067811865476, 0, 0],
                                       name='custom')
else:
    pd['init_wavelet'] = None

if pd['cell'] == 'GRU':
    cell = GRUCell(input_size, pd['hidden'], output_size).cuda()
elif pd['cell'] == 'WaveletGRU':
    cell = WaveletGRU(input_size, pd['hidden'], output_size,
                      mode=pd['compression_mode']).cuda()
elif pd['cell'] == 'FastFoodGRU':
    cell = FastFoodGRU(input_size, pd['hidden'], output_size).cuda()
else:
    raise NotImplementedError()


for e in range(pd['epochs']):
    # go for spectrum prediction here!!
    train_keys, train_features, train_phones, train_feat_len, train_phone_len = timit.get_train_batches()
