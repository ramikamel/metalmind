"""Campaign-frozen numerical policies and bounded input domains."""
from metalmind.contracts import Contract

class CorrectnessPolicy(Contract):
    atol: float
    rtol: float
    max_input: float
    max_parameter: float | None
    exclude_subnormals: bool = True

def policy_for(workload):
    reduction=workload.family!="bias_gelu"
    half=workload.inputs[0].dtype=="float16"
    atol,rtol=((2e-3,2e-2) if reduction else (1e-3,1e-2)) if half else ((1e-5,1e-4) if reduction else (1e-6,1e-5))
    bound={"bias_gelu":10.,"rmsnorm":100.,"softmax":80.}[workload.family]
    return CorrectnessPolicy(atol=atol,rtol=rtol,max_input=bound,max_parameter=None if workload.family=="softmax" else 10.)
