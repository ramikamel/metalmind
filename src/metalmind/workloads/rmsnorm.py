import torch

def reference(x,weight):
    z=x.float()
    return (z*torch.rsqrt(z.square().mean(dim=-1,keepdim=True)+1e-5)*weight.float()).to(x.dtype)

def oracle(x,weight):
    z=x.double()
    return z/torch.sqrt(z.square().sum(dim=-1,keepdim=True)/z.shape[-1]+1e-5)*weight.double()
