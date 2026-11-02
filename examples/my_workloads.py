import torch
import torch.nn.functional as F


def bias_gelu(x,bias):
    return F.gelu(x+bias,approximate="tanh")


def exact_gelu(x,bias):
    return F.gelu(x+bias,approximate="none")


def row_softmax(x):
    return torch.softmax(x,dim=-1)


def rmsnorm(x,weight):
    return x*torch.rsqrt(x.square().mean(dim=-1,keepdim=True)+1e-5)*weight
