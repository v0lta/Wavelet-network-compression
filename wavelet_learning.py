import pywt
import pywt.data
import numpy as np
import matplotlib.pyplot as plt


def format_array(a):
    """Consistent array representation across different systems"""
    a = np.where(np.abs(a) < 1e-5, 0, a)
    return np.array2string(a, precision=5, separator=' ', suppress_small=True)


x = [1, 2, 1, 5, -1, 8, 4, 6]
cA, cD = pywt.dwt(x, 'db2')

print(cA)
print(cD)

# Load image
original = pywt.data.camera()

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
