"""Offline module demonstration; no model key or generation request is used."""
import argparse
import json
from pathlib import Path
import torch
import metalmind
from examples.module_workloads import make_demo
from metalmind.artifacts.store import atomic_json
from metalmind.graph.rewrite import ModuleOptimizationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact",required=True,type=Path)
    parser.add_argument("--assume-domain",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=Path("runs/modules"))
    args = parser.parse_args()
    module,inputs = make_demo()
    result = metalmind.optimize_module(module,inputs,config=ModuleOptimizationConfig(
        artifact_paths=(args.artifact,),domain_mode="assume" if args.assume_domain else "checked",output_dir=args.output_dir))
    x = inputs[0]
    with torch.inference_mode():
        new = x*.5
        torch.testing.assert_close(result.module(new),module(new),atol=1e-5,rtol=1e-4)
        module.bias.add_(.05)
        torch.testing.assert_close(result.module(new),module(new),atol=1e-5,rtol=1e-4)
        changed = x[:2].contiguous()
        torch.testing.assert_close(result.module(changed),module(changed))
    grad_input = x.detach().requires_grad_(True)
    result.module(grad_input).sum().backward()
    assert grad_input.grad is not None and module.bias.grad is not None
    checks = {"new_values_match":True,"updated_parameter_values_match":True,
        "changed_shape_fallback":True,"gradient_fallback":True,"optimized_enabled":result.enabled}
    atomic_json(result.run_path/"post-run-checks.json",checks)
    print(json.dumps({**result.report(),"post_run_checks":checks},indent=2))


if __name__ == "__main__":
    main()
