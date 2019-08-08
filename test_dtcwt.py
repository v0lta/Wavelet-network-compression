import sys
sys.path.append('./pytorch_wavelets-master')
import torch
from pytorch_wavelets import DTCWTForward, DTCWTInverse


xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b').cuda()
X = torch.randn(10, 5, 64, 64).cuda()
Yl, Yh = xfm(X)
ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').cuda()
Y = ifm((Yl, Yh))
print(torch.norm(torch.abs(X - Y)))
