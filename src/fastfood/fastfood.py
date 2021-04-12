# Created by moritz (wolter@cs.uni-bonn.de)
"""
A fastfood layer implementation.
"""
import torch
import numpy as np
from torch.nn.parameter import Parameter
from src.fastfood.fwht import matmul_wht as wht
from scipy.linalg import hadamard


def diag_mul(vector, mat):
    return torch.mm(torch.diag(vector), mat)


class FastFoodLayer(torch.nn.Module):
    """
    Create a learn-able Fast-food layer as described in
    https://arxiv.org/abs/1412.7149
    The weights are parametrized by S*H*G*P*H*B
    With S,G,B diagonal matrices, P a random permutation and H the Walsh-Hadamard transform.
    """
    def __init__(self, depth, p_drop=0.5):
        super().__init__()
        print('fastfood dropout:', p_drop)
        ones = np.ones(depth, np.float32)
        self.diag_vec_s = Parameter(torch.from_numpy(ones))
        self.diag_vec_g = Parameter(torch.from_numpy(ones))
        self.diag_vec_b = Parameter(torch.from_numpy(ones))
        perm = np.random.permutation(np.eye(depth, dtype=np.float32))
        self.perm = Parameter(torch.from_numpy(perm), requires_grad=False)
        self.depth = depth
        self.drop_s = torch.nn.Dropout(p=p_drop)
        self.drop_g = torch.nn.Dropout(p=p_drop)
        self.drop_b = torch.nn.Dropout(p=p_drop)
        self.h_mat = Parameter(torch.from_numpy(hadamard(depth).astype(np.float32)), requires_grad=False)

    def mul_s(self, x):
        return torch.mm(x, self.drop_s(torch.diag(self.diag_vec_s)))

    def mul_g(self, x):
        return torch.mm(x, self.drop_g(torch.diag(self.diag_vec_g)))

    def mul_b(self, x):
        return torch.mm(x, self.drop_b(torch.diag(self.diag_vec_b)))

    def mul_p(self, x):
        return torch.mm(x, self.perm)

    def forward(self, x):
        return self.mul_s(wht(self.mul_g(self.mul_p(wht(self.mul_b(x),
                                                        h_mat=self.h_mat))),
                              h_mat=self.h_mat, inverse=True))

    def extra_repr(self):
        return 'depth={}'.format(self.depth)
