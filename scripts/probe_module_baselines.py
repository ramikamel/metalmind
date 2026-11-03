"""Sequential whole-module baseline diagnostic. No model requests."""
import argparse
import json
from pathlib import Path
import secrets
import shutil
import time
import uuid
import torch
from examples.module_workloads import make_demo
from metalmind.artifacts.store import atomic_json
from metalmind.artifacts.export import environment_fingerprint, _hardware_profile
from metalmind.benchmarking.timer import measure_pair, BenchmarkUnstable
from metalmind.benchmarking.statistics import summarize_pairs
from metalmind.graph.capture import capture_module
from metalmind.graph.profile import assert_measurement_ready
from metalmind.graph.rewrite import ModuleOptimizationConfig, GuardedModule, _select_artifacts, _rewrite_graph, _verify_private, _compare, _versions
from metalmind.runtime.lock import gpu_lock
from metalmind.runtime.worker import _record_fault


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=Path("runs/module-baselines"))
    args=parser.parse_args()
    folder=args.output_dir/uuid.uuid4().hex;folder.mkdir(parents=True,exist_ok=False)
    report={"schema_version":1,"phase":"setup","model_requests":0,"domain_mode":"assume",
        "factory":"examples.module_workloads:make_demo","timing_boundary":"synchronized whole module including guards, unchanged operations and allocation",
        "environment_fingerprint":environment_fingerprint(),"hardware":_hardware_profile(),"sessions":[],"unavailable":[]}
    atomic_json(folder/"report.json",report)
    shutil.copy2(__file__,folder/"diagnostic.py")
    shutil.copy2(Path(__file__).resolve().parents[1]/"examples/module_workloads.py",folder/"module_workloads.py")
    assert_measurement_ready()
    module,inputs=make_demo();captured=capture_module(module,inputs)
    config=ModuleOptimizationConfig(artifact_paths=(args.artifact,),domain_mode="assume")
    with gpu_lock():
        selected,identities,issues=_select_artifacts(captured,config)
        if not selected: raise ValueError("No compatible promoted artifact: " + str(issues))
        candidate=GuardedModule(module,_rewrite_graph(captured,selected),captured.guard)
        report.update({"graph_hash":captured.graph_hash,"artifact_ids":identities,"shape":list(inputs[0].shape),"dtype":"float32"})
        seeds=tuple(secrets.randbits(63) for _ in range(8));atomic_json(folder/"private-fixtures.json",{"seeds":seeds})
        report["candidate_verification"]=_verify_private(captured,candidate,selected,config,seeds)
        started=time.monotonic()
        try:
            compiled=torch.compile(module,fullgraph=True)
            maximum=0.
            with torch.inference_mode():
                for seed in seeds:
                    rng=torch.Generator().manual_seed(seed)
                    x=(torch.randn(inputs[0].shape,generator=rng)*.25).clamp_(-1,1).to("mps")
                    versions=_versions((x,*module.parameters()))
                    actual=compiled(x);torch.mps.synchronize()
                    if actual.device.type!="mps": raise ValueError("Compiled output left MPS")
                    maximum=max(maximum,_compare(actual,module(x)))
                    if versions!=_versions((x,*module.parameters())): raise ValueError("Compiled baseline mutated input/state")
                saved=module.bias.clone()
                try:
                    module.bias.add_(.05)
                    maximum=max(maximum,_compare(compiled(*inputs),module(*inputs)))
                    _compare(candidate(*inputs),module(*inputs))
                finally: module.bias.copy_(saved)
            report["compiled_verification"]={"passed":True,"cases":8,"parameter_update_case":True,"max_absolute_error":maximum}
        except Exception as exc:
            compiled=None
            report["unavailable"].append("torch_compile_inductor_fullgraph: "+type(exc).__name__+": "+str(exc)[:600])
        report["compiled_setup_seconds"]=time.monotonic()-started
        report["phase"]="measuring";atomic_json(folder/"report.json",report)
        baselines=[("original_eager_whole_module",module)]
        if compiled is not None: baselines.append(("torch_compile_inductor_fullgraph_whole_module",compiled))
        try:
            for session in range(2):
                if session: time.sleep(60)
                results=[]
                for index,(name,baseline) in enumerate(baselines):
                    seed=211+session*2+index
                    try:
                        series=measure_pair(lambda:baseline(*inputs),lambda:candidate(*inputs),pairs=30,seed=seed)
                        summary=summarize_pairs(series.baseline_ms,series.candidate_ms,seed=seed,resamples=10000)
                        results.append({"baseline":name,"status":"valid","seed":seed,"bootstrap_resamples":10000,
                            "series":series.model_dump(mode="json"),"summary":summary.model_dump(mode="json")})
                    except BenchmarkUnstable as exc: results.append({"baseline":name,"status":"inconclusive","reason":str(exc)})
                with torch.inference_mode():
                    if not captured.guard.matches(inputs): raise ValueError("Module state changed during measurements")
                report["sessions"].append({"session":session+1,"comparisons":results})
                atomic_json(folder/"report.json",report)
        except RuntimeError as exc:
            _record_fault(str(exc));report["phase"]="failed";report["reason"]=str(exc)[:600];atomic_json(folder/"report.json",report);raise
        report["phase"]="terminal"
        compiled_results=[c for s in report["sessions"] for c in s["comparisons"] if c["baseline"].startswith("torch_compile")]
        report["compiled_comparison_outcome"]=("unavailable" if compiled is None else "inconclusive" if any(c["status"]!="valid" for c in compiled_results)
            else "confirmed_gain" if all(c["summary"]["speedup"]>=1.05 and c["summary"]["lower"]>1 for c in compiled_results) else "no_confirmed_gain")
        atomic_json(folder/"report.json",report)
    print(json.dumps({"run_path":str(folder),"compiled_comparison_outcome":report["compiled_comparison_outcome"],
        "comparisons":[{"session":s["session"],"results":[{k:v for k,v in c.items() if k!="series"} for c in s["comparisons"]]} for s in report["sessions"]],"unavailable":report["unavailable"],"model_requests":0},indent=2))


if __name__=="__main__": main()
