"""Run from the repo root with python -m examples.optimize_function."""
import torch
import metalmind
from metalmind.config import RunConfig
from examples.my_workloads import bias_gelu

if __name__=='__main__':
    x=torch.randn(256,1024,device='mps');bias=torch.randn(1024,device='mps')
    analysis=metalmind.analyze(bias_gelu,(x,bias))
    if not analysis.supported: raise SystemExit(analysis.reason)
    result=metalmind.optimize(analysis.workload,config=RunConfig(max_candidates=2,max_requests=2))
    optimized=result.build_callable(fallback=bias_gelu)
    torch.testing.assert_close(optimized(x,bias),bias_gelu(x,bias),atol=1e-6,rtol=1e-5)
    # New values with the same signature use a promoted artifact when available.
    y=optimized(x*.5,bias*.25)
    torch.testing.assert_close(y,bias_gelu(x*.5,bias*.25),atol=1e-6,rtol=1e-5)
    # A changed shape uses the original function.
    changed=torch.randn(2,1024,device='mps')
    torch.testing.assert_close(optimized(changed,bias),bias_gelu(changed,bias))
    print(result)
    print("Optimized artifact enabled:",optimized.optimized is not None)
