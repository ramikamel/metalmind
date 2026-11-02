"""Local MetalMind diagnostics, optimization, evidence, and campaign commands."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from metalmind.runtime.environment import probe_environment


def _function(name):
    import importlib
    import sys
    # Explicit user-selected local module; never a model-generated import path.
    if str(Path.cwd()) not in sys.path: sys.path.insert(0,str(Path.cwd()))
    module,attribute=name.split(":",1)
    return getattr(importlib.import_module(module),attribute)


def main(argv=None):
    parser=argparse.ArgumentParser(prog="metalmind")
    commands=parser.add_subparsers(dest="command",required=True)
    doctor=commands.add_parser("doctor",help="Describe the local PyTorch/Metal environment")
    doctor.add_argument("--json",action="store_true")
    doctor.add_argument("--smoke",action="store_true")
    for name in ("analyze","optimize"):
        cmd=commands.add_parser(name)
        cmd.add_argument("--workload",choices=("bias_gelu","rmsnorm","softmax"),default="bias_gelu")
        cmd.add_argument("--case",choices=("small","medium","large"),default="medium")
        cmd.add_argument("--dtype",choices=("float32","float16"),default="float32")
        cmd.add_argument("--function",help="Trusted importable module:function")
        cmd.add_argument("--json",action="store_true")
        if name=="optimize":
            cmd.add_argument("--config",type=Path)
            cmd.add_argument("--max-requests",type=int)
            cmd.add_argument("--max-candidates",type=int)
            cmd.add_argument("--assume-domain",action="store_true",help="Explicitly guarantee the documented numerical input domain")
    report=commands.add_parser("report");report.add_argument("--run",type=Path,required=True)
    resume=commands.add_parser("resume");resume.add_argument("--run",type=Path,required=True)
    for name in ("verify","revalidate"):
        cmd=commands.add_parser(name);cmd.add_argument("--artifact",type=Path,required=True)
        cmd.add_argument("--function",required=name=="revalidate")
        if name=="revalidate": cmd.add_argument("--assume-domain",action="store_true")
    campaign=commands.add_parser("evaluate")
    campaign.add_argument("--suite",type=Path,required=True)
    campaign.add_argument("--execute",action="store_true",help="Run the full configured campaign, including billable model requests")
    args=parser.parse_args(argv)
    if args.command=="doctor": return _doctor(args)
    try: return _run(args)
    except (ValueError,OSError,RuntimeError) as exc:
        print(json.dumps({"outcome":"failed","reason":str(exc)[:1000]}));return 1


def _doctor(args):
    info = probe_environment()
    report = asdict(info)
    ready = info.capabilities.mps_available and info.capabilities.compile_shader
    if args.smoke:
        report["shader_health_passed"] = False
        if ready:
            from metalmind.runtime.health import check_health
            report["shader_health_passed"] = check_health()
            if report["shader_health_passed"]:
                from metalmind.runtime.worker import clear_device_fault
                clear_device_fault()
        ready = ready and report["shader_health_passed"]
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(f"Python {info.python_version} | PyTorch {info.torch_version} | {info.architecture}")
        print(f"MPS available: {info.capabilities.mps_available}")
        print(f"Shader API available: {info.capabilities.compile_shader}")
        if args.smoke:
            print(f"Shader health passed: {report['shader_health_passed']}")
        if info.unavailable_reason:
            print(info.unavailable_reason)
    return 0 if ready else 1


def _run(args):
    from metalmind.config import RunConfig,load_config
    from metalmind.workloads.registry import get_workload
    from metalmind.artifacts.export import finalize_run,verify_artifact_evidence
    from metalmind.artifacts.store import atomic_json
    if args.command in ("analyze","optimize"):
        workload=get_workload(args.workload,args.case,args.dtype)
        if args.function:
            from metalmind.analysis import analyze
            from metalmind.workloads.fixtures import make_public_fixtures
            result=analyze(_function(args.function),make_public_fixtures(workload)[0].inputs)
            if not result.supported: print(result.model_dump_json(indent=2));return 1
            workload=result.workload
        if args.command=="analyze":
            print(json.dumps({"supported":True,"workload":workload.model_dump(mode="json"),"hypothesis":"Fusion may reduce traffic and dispatch overhead; no performance measurement yet."},indent=2));return 0
        config=load_config(args.config)
        patch={}
        if args.max_requests is not None: patch['max_requests']=args.max_requests
        if args.max_candidates is not None: patch['max_candidates']=args.max_candidates
        if args.assume_domain: patch['domain_mode']='assume'
        config=RunConfig.model_validate(config.model_dump()|patch)
        from metalmind.api import optimize
        result=optimize(workload,config=config)
        from metalmind.reporting.summary import build_run_summary
        from metalmind.reporting.html import render_report
        render_report(build_run_summary(result.run_path),result.run_path/'report.html')
        print(json.dumps(asdict(result),default=str,indent=2))
        return 1 if result.outcome=='failed' else 0
    if args.command=='report':
        from metalmind.reporting.summary import build_run_summary
        from metalmind.reporting.html import render_report
        print(render_report(build_run_summary(args.run),args.run/'report.html'));return 0
    if args.command=='resume':
        from metalmind.agent.controller import SearchController
        result=finalize_run(SearchController().resume(args.run))
        print(json.dumps(asdict(result),default=str,indent=2));return 0
    if args.command=='verify':
        import secrets
        from metalmind.contracts import EvaluationRequest
        from metalmind.runtime.worker import evaluate
        manifest=verify_artifact_evidence(args.artifact)
        if manifest.workload.reference_id.startswith('python:'):
            if not args.function: raise ValueError('Auditing a custom reference requires explicit --function module:name')
            from metalmind.workloads.capture import describe_reference
            if describe_reference(_function(args.function))!=manifest.workload.reference_id:
                raise ValueError('Explicit reference does not match artifact')
        from metalmind.artifacts.store import ArtifactStore
        from metalmind.runtime.environment import probe_environment
        from metalmind.verification.final_gate import run_final_gate
        audit=ArtifactStore.create_run(Path('runs/audits'),RunConfig(),probe_environment())
        result=run_final_gate(manifest.candidate,manifest.workload,tuple(secrets.randbits(63) for _ in range(16)),marker_path=audit.path/'private-gate-consumed',timeout=120)
        atomic_json(audit.path/'verification.json',result.model_dump(mode='json'))
        print(result.model_dump_json(indent=2));return 0 if result.status=='valid' else 1
    if args.command=='revalidate':
        from metalmind.artifacts.loader import revalidate_artifact
        result=revalidate_artifact(args.artifact,fallback=_function(args.function),config=RunConfig(domain_mode="assume" if args.assume_domain else "checked"))
        print(json.dumps(asdict(result),default=str,indent=2));return 0
    if args.command=='evaluate':
        import yaml
        from metalmind.evaluation import CampaignConfig,freeze_campaign,run_campaign
        config=CampaignConfig.model_validate(yaml.safe_load(args.suite.read_text()))
        report=run_campaign(config) if args.execute else freeze_campaign(config)
        print(json.dumps(report,indent=2));return 0
    raise ValueError('unknown command')

if __name__=='__main__': raise SystemExit(main())
