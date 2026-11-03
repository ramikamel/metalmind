"""Whole-module compiler baselines must validate before they enter promotion."""
import torch
from metalmind.graph.verification import _compare, _versions, make_private_inputs


def collect_module_baselines(module, example_inputs, seeds):
    baselines=[("original_eager_whole_module",module)]
    unavailable=[]
    try:
        compiled=torch.compile(module,fullgraph=True)
        parameters=tuple(module.parameters())+tuple(module.buffers())
        for seed in seeds:
            inputs=make_private_inputs(example_inputs,seed)
            before=_versions((*inputs,*parameters))
            with torch.inference_mode():
                actual=compiled(*inputs);torch.mps.synchronize()
                if not isinstance(actual,torch.Tensor) or actual.device.type!="mps":
                    # Nested tensor outputs use the same recursive comparison below.
                    def tensors(value):
                        if isinstance(value,torch.Tensor): return [value]
                        if isinstance(value,dict): return [t for v in value.values() for t in tensors(v)]
                        if isinstance(value,(tuple,list)): return [t for v in value for t in tensors(v)]
                        return []
                    leaves=tensors(actual)
                    if not leaves or any(t.device.type!="mps" for t in leaves): raise ValueError("Compiled outputs must remain on MPS")
                _compare(actual,module(*inputs))
            if before!=_versions((*inputs,*parameters)):
                raise ValueError("Compiled module mutated inputs or registered tensors")
        baselines.append(("torch_compile_inductor_fullgraph_whole_module",compiled))
    except Exception as exc:
        unavailable.append("torch_compile_inductor_fullgraph_whole_module: "+type(exc).__name__+": "+str(exc)[:500])
    return baselines,tuple(unavailable)
