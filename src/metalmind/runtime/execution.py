"""The shared public-call boundary used both for benchmarks and exports."""
import torch
from metalmind.runtime.abi import matches_signature, bind_kernel_args
from metalmind.verification.numerics import inputs_in_domain

def build_execution_callable(workload,kernel,*,fallback,domain_mode):
    if domain_mode not in ("checked","assume"): raise ValueError("unknown domain mode")
    def call(*inputs):
        if not matches_signature(workload,inputs): return fallback(*inputs)
        if domain_mode=="checked" and not inputs_in_domain(workload,inputs): return fallback(*inputs)
        out=torch.empty(workload.output.shape,device="mps",dtype=getattr(torch,workload.output.dtype))
        kernel.function(*bind_kernel_args(workload,inputs,out),threads=call.proposal.threads,group_size=call.proposal.group_size)
        return out
    # Attached by the trusted caller; never generated host Python.
    call.proposal=None
    return call
