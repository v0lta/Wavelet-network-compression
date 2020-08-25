# wavelet-network-compression 
This repository implements a learnable fast wavelet transform for use in machine learning models
with pytorch.
It also contains the source code used to create the experimental results which are reportet
in the paper Neural network compression via learnable wavelet transforms which is 
available here https://arxiv.org/pdf/2004.09569.pdf.
The most interesting modules are `wavelet_learning/wavelet_linear.py`
and `wavelet_learning/learn_wave.py` , it's where the wavelet
optimization happens. 
When using this code please never forget to add the wavelet loss term
to the cost. See the class Net from the `mnist_compression.py` file
for an example on how to do that.

A re-implementation of the Hadamard-Transform based layer as described in
https://arxiv.org/abs/1412.7149 can be found in the 
`fastfood/fastfood.py` module.

###### Experiments:
To repeat the experiments from the paper run any of the run_all files in python.
Overall the performace is equivalent to to other state of the art approaches,
like the Hadamard transfrom based layer, with extra flexibility.

###### Citation:
A preprint is available at: https://arxiv.org/pdf/2004.09569.pdf. If you find this work useful please consider citing the paper:
```
@inproceedings{wolter2020neural,
  author={Wolter, Moritz and Lin, Shaohui and Yao, Angela},
  title={Neural Network compression via learnable wavelet transforms},
  booktitle={29th International Conference on Artificial Neural Networks},
  year = {2020}
}
```

###### Future Work:
The current implementation is probably not as efficient as possible. I am hoping that 
this project will inspire work on highly optimized GPU-FWT implementations.


###### Funding:
This work has been funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) YA 447/2-1 (FOR2535 Anticipating Human Behavior)
as well as by the National Research Foundation of Singapore under its NRF Fellowship Programme [NRF-NRFFAI1-2019-0001].
