import torch

def reference(x):
    return torch.softmax(x.float(),dim=-1).to(x.dtype)

def oracle(x):
    z=x.double(); shifted=z-z.max(dim=-1,keepdim=True).values
    numerator=shifted.exp()
    return numerator/numerator.sum(dim=-1,keepdim=True)
