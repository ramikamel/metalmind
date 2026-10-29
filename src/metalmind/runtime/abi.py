"""Host-owned ABI version 1: inputs, output, uint32 B/D, FP32 epsilon."""
import hashlib
import json
import torch
from metalmind.contracts import WorkloadSpec

def signature_for(workload: WorkloadSpec) -> str:
    canonical=json.dumps(workload.model_dump(mode="json"),sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()

def _matches(spec,tensor):
    return (isinstance(tensor,torch.Tensor) and tuple(tensor.shape)==spec.shape
        and tuple(tensor.stride())==spec.strides and str(tensor.dtype)=="torch."+spec.dtype
        and tensor.device.type==spec.device and tensor.storage_offset()==spec.storage_offset
        and tensor.is_contiguous() and tensor.storage_offset()==0)

def matches_signature(workload, inputs) -> bool:
    if len(inputs)!=len(workload.inputs) or not all(_matches(s,t) for s,t in zip(workload.inputs,inputs)):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in inputs):
        return False
    pointers=[t.untyped_storage().data_ptr() for t in inputs]
    return len(set(pointers))==len(pointers)

def bind_kernel_args(workload, inputs, output) -> tuple:
    if not matches_signature(workload,inputs) or not _matches(workload.output,output):
        raise ValueError("Buffers do not match workload ABI")
    if output.untyped_storage().data_ptr() in [t.untyped_storage().data_ptr() for t in inputs]:
        raise ValueError("Output cannot alias inputs")
    b,d=workload.inputs[0].shape
    return (*inputs,output,b,d,float(workload.epsilon))
