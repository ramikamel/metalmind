"""Small pure inference modules for the explicitly scoped graph demonstration."""
import torch
import torch.nn.functional as F

class ResidualActivation(torch.nn.Module):
    def __init__(self, width=1024):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.linspace(-.05,.05,width))
        self.residual_scale = .25

    def forward(self, x):
        activation = F.gelu(x + self.bias, approximate="tanh")
        return activation + x * self.residual_scale


def make_demo():
    module = ResidualActivation().eval().to("mps")
    x = torch.empty(256,1024,device="mps").uniform_(-1,1)
    return module, (x,)


def make_cpu_demo():
    return ResidualActivation(3).eval(), (torch.ones(2,3),)
