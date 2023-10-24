import torch
import torch.nn as nn


class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):

        # b, c, h, w = x.size()
        #
        # n = w * h - 1
        #
        # x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        # y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        #
        # return x * self.activaton(y)

        b, c, d, h, w = x.size()

        n = d * h * w - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

if __name__ == '__main__':
    model = simam_module()
    x= torch.randn(1,3,3,64,64)
    y=model(x)
    print(y.size())
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        model(x)

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5))