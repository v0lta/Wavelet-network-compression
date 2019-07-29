import numpy as np
import torch
import matplotlib.pyplot as plt


# class ClassicIkedaMap:
#     def __init__(self, a, b, c):
#         self.a = a
#         self.a = b
#         self.c = c

#     def __call__(z):

#         def exp(z):
#             r = torch.cos(z)
#             i = torch.sin(z)
#             return torch.cat([r, i], dim=-1)

#         def abs(z):
#             return torch.sqrt(z[..., 0]*z[..., 0] + z[..., 1]*z[..., 1])

#     return self.a + self.b*exp(abs(z) + self.c)


class IkedaMap:
    def __init__(self, u):
        self.u = u

    def __call__(self, z):
        x = z[0]
        y = z[1]
        tn = .4 - 6./(1 + x*x + y*y)
        xn = 1 + self.u*(x*torch.cos(tn) - y*torch.sin(tn))
        yn = self.u*(x*torch.sin(tn) + y*torch.cos(tn))
        return torch.stack([xn, yn], -1)


if __name__ == "__main__":
    ikeda = IkedaMap(.918)

    trajectories = 150
    tsteps = 500

    t_lst = []
    for _ in range(trajectories):
        y = [torch.rand(2)*30.-15.]
        for t in range(tsteps):
            y.append(ikeda(y[-1]))

        y = torch.stack(y)
        t_lst.append(y)

    for t in t_lst:
        y_np = t.detach().cpu().numpy()
        plt.plot(y_np[:, 0],  y_np[:, 1])

    plt.show()