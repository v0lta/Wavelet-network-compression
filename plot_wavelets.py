import matplotlib.pyplot as plt
import sys
sys.path.append('./pytorch_wavelets-master')
import torch
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from pytorch_wavelets.dtcwt.coeffs import biort

out = biort('near_sym_a')
out0 = out[0]
out1 = out[1]
out2 = out[2]
out3 = out[3]

plt.plot(out0)
plt.show()
plt.plot(out1)
plt.show()
plt.plot(out2)
plt.show()
plt.plot(out3)
plt.show()
