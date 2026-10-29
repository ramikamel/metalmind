import math
import torch
import torch.nn.functional as F

def reference(x,bias):
    return F.gelu(x.float()+bias.float(),approximate="tanh").to(x.dtype)

def oracle(x,bias):
    z=x.double()+bias.double()
    return .5*z*(1+torch.tanh(math.sqrt(2/math.pi)*(z+.044715*z**3)))
