# Wavelet Network Compression 
This repository implements a learnable fast wavelet transform for use in machine learning models
with PyTorch.

It also contains the source code used to create the experimental results,
as reported in the paper Neural network compression via learnable wavelet transforms. A preprint is
available here https://arxiv.org/pdf/2004.09569.pdf the Springer-version 
at https://link.springer.com/chapter/10.1007/978-3-030-61616-8_4 .

The most relevant modules are `src/wavelet_learning/wavelet_linear.py`
and `src/wavelet_learning/learn_wave.py` it's where the wavelet
optimization happens. 
When using this code, please never forget to add the wavelet loss term
to the cost. See the class Net from the `mnist_compression.py` file
for an example of how to do that.

The `src/fastfood/fastfood.py` module re-implements the Hadamard-Transform 
based layer as described in https://arxiv.org/abs/1412.7149 .

###### Experiments:
To repeat the experiments from the paper, run any of the run_all files 
from the scripts folder in python.
Running ```$ python scripts/run_all_mnist.py ``` for example, repeats the MNIST
CNN experiments from the paper.
Overall the performance is equivalent to state-of-the-art approaches,
like the Hadamard transform based layer, with extra flexibility.

###### Citation:
If you find this work useful please consider citing the paper:
```
@inproceedings{wolter2020neural,
  author={Wolter, Moritz and Lin, Shaohui and Yao, Angela},
  title={Neural Network compression via learnable wavelet transforms},
  booktitle={29th International Conference on Artificial Neural Networks},
  year = {2020}
}
```

###### Toolbox:
The current version of the PyTorch-Wavelet-Toolbox is available at https://github.com/v0lta/PyTorch-Wavelet-Toolbox .

###### Funding:
This work has been funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) YA 447/2-1 (FOR2535 Anticipating Human Behavior)
as well as by the National Research Foundation of Singapore under its NRF Fellowship Programme [NRF-NRFFAI1-2019-0001].
