"""Importable FP16 functions with explicit FP32 intermediate arithmetic."""
import torch
import torch.nn.functional as F


def bias_gelu(x, bias):
    return F.gelu(x.to(dtype=torch.float32) + bias.float(), approximate="tanh").half()


def rmsnorm(x, weight):
    z = x.float()
    return (z * torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + 1e-5) * weight.float()).to(torch.float16)


def row_softmax(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)
