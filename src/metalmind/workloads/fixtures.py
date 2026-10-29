"""CPU-generated public and private fixture sets; never prompt inputs."""
from dataclasses import dataclass
import torch
from metalmind.verification.policy import policy_for

@dataclass(frozen=True)
class Fixture:
    category: str
    inputs: tuple[torch.Tensor,...]

def _fixtures(workload,seeds):
    dtype=getattr(torch,workload.inputs[0].dtype)
    policy=policy_for(workload)
    fixtures=[]
    for seed in seeds:
        generator=torch.Generator().manual_seed(seed)
        tensors=[]
        for i,spec in enumerate(workload.inputs):
            bound=policy.max_input if i==0 else policy.max_parameter
            t=(torch.rand(spec.shape,generator=generator)*2-1)*min(bound,3.)
            t=t.to(dtype)
            t[torch.abs(t)<torch.finfo(dtype).tiny]=0
            tensors.append(t)
        fixtures.append(Fixture("random",tuple(tensors)))
    for category in ("zeros","ones","alternating","near_zero","bounds","coded"):
        tensors=[]
        for i,spec in enumerate(workload.inputs):
            bound=policy.max_input if i==0 else policy.max_parameter
            n=1
            for size in spec.shape: n*=size
            indices=torch.arange(n,dtype=torch.float64).reshape(spec.shape)
            if category=="zeros": t=torch.zeros(spec.shape)
            elif category=="ones": t=torch.ones(spec.shape)
            elif category=="alternating": t=(indices.remainder(2)*2-1)
            elif category=="near_zero": t=(indices.remainder(3)-1)*1e-3
            elif category=="bounds": t=(indices.remainder(2)*2-1)*bound
            else: t=(indices.remainder(17)-8)/8
            tensors.append(t.to(dtype))
        fixtures.append(Fixture(category,tuple(tensors)))
    return fixtures

def make_public_fixtures(workload):
    return _fixtures(workload,(11,29,47,71))

def make_private_fixtures(workload,private_seed_set):
    if len(private_seed_set)!=16 or len(set(private_seed_set))!=16:
        raise ValueError("Final verification requires sixteen distinct private seeds")
    return _fixtures(workload,private_seed_set)
