"""Trusted references and their exact supported signature matrix."""
import hashlib
import inspect
import importlib
import json
import dis
from pathlib import Path
from metalmind.contracts import TensorSpec, WorkloadSpec
from metalmind.workloads import bias_gelu, rmsnorm, softmax

_MODULES={"bias_gelu":bias_gelu,"rmsnorm":rmsnorm,"softmax":softmax}
_CASES={"small":(64,257),"medium":(256,1024),"large":(512,4096)}

def reference_hash(family):
    return hashlib.sha256(inspect.getsource(_MODULES[family]).encode()).hexdigest()

def get_workload(workload_id,case_id,dtype="float32"):
    b,d=_CASES[case_id]
    if workload_id=="softmax" and case_id=="large": b=128
    return workload_for_shape(workload_id,(b,d),dtype=dtype,case_id=case_id)

def workload_for_shape(family,shape,*,dtype="float32",case_id="custom"):
    b,d=shape
    main=TensorSpec(shape=(b,d),strides=(d,1),dtype=dtype)
    inputs=(main,) if family=="softmax" else (main,TensorSpec(shape=(d,),strides=(1,),dtype=dtype))
    return WorkloadSpec(workload_id=f"{family}:{case_id}",family=family,inputs=inputs,output=main,
        epsilon=1e-5 if family=="rmsnorm" else 0.,reference_id=family,
        reference_hash=reference_hash(family))

def resolve_reference(reference_id):
    if reference_id.startswith("python:"):
        _,module,name=reference_id.split(":")
        return getattr(importlib.import_module(module),name)
    try: return _MODULES[reference_id].reference
    except KeyError: raise ValueError("Unregistered reference identifier") from None

def resolve_oracle(family):
    return _MODULES[family].oracle


def fingerprint_reference(reference_id):
    if not reference_id.startswith("python:"): return reference_hash(reference_id)
    fn=resolve_reference(reference_id)
    from metalmind.workloads.capture import describe_reference
    describe_reference(fn)
    module=importlib.import_module(fn.__module__)
    scalars={name:fn.__globals__.get(name) for name in {i.argval for i in dis.get_instructions(fn) if i.opname=='LOAD_GLOBAL'}
        if isinstance(fn.__globals__.get(name),(str,int,float,bool))}
    identity=Path(module.__file__).read_bytes()+json.dumps([fn.__defaults__,scalars],sort_keys=True).encode()
    return hashlib.sha256(identity).hexdigest()
