"""Evaluate a trusted fusion of the value check into the saved vectorized GELU."""
from dataclasses import asdict
import argparse
from pathlib import Path
import runpy

import torch

from metalmind.artifacts.export import _hardware_profile, verify_artifact_evidence
from metalmind.artifacts.store import atomic_json
from metalmind.benchmarking.baselines import collect_baselines
from metalmind.runtime.abi import matches_signature
from metalmind.runtime.environment import probe_environment
from metalmind.runtime.execution import build_execution_callable
from metalmind.runtime.lock import gpu_lock
from metalmind.runtime.metal import MetalBackend
from metalmind.verification.numerics import compare_outputs, inputs_in_domain
from metalmind.verification.policy import policy_for
from metalmind.workloads.fixtures import make_public_fixtures
from metalmind.workloads.registry import resolve_reference

def make_source(source):
    abi = 'constant float& epsilon, uint2 gid'
    load = '    float4 z = xv + pv;'
    assert source.count(abi) == source.count(load) == 1
    source = source.replace(abi, 'constant float& epsilon, device atomic_uint* rejected, uint2 gid')
    body = """    uint4 xm = as_type<uint4>(xv) & 0x7fffffffu;
    uint4 pm = as_type<uint4>(pv) & 0x7fffffffu;
    bool bad_x = any(xm > 0x41200000u) || any((xm != 0u) & (xm < 0x00800000u));
    bool bad_p = any(pm > 0x41200000u) || any((pm != 0u) & (pm < 0x00800000u));
    if (bad_x || bad_p) {
        atomic_store_explicit(rejected, 1u, memory_order_relaxed);
        return;
    }
"""
    return source.replace(load,body+load)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('output', type=Path)
    args=parser.parse_args()
    pair=runpy.run_path(str(Path(__file__).with_name('procedure.py')))['pair']
    manifest=verify_artifact_evidence(args.artifact)
    workload=manifest.workload
    assert workload.family == 'bias_gelu' and workload.inputs[0].shape == (512,4096)
    assert workload.inputs[0].dtype == 'float32'
    proposal=manifest.candidate.proposal
    policy=policy_for(workload)
    assert policy.max_input == policy.max_parameter == 10
    reference=resolve_reference(workload.reference_id)
    fallback_count=0

    def fallback(*args):
        nonlocal fallback_count
        fallback_count+=1
        return reference(*args)

    source=make_source(proposal.source)
    args.output.with_suffix('.metal').write_text(source)
    with gpu_lock(timeout=70), torch.inference_mode():
        backend=MetalBackend()
        checked_kernel=backend.compile(source,entry_point='metalmind_kernel')
        standard=build_execution_callable(workload,backend.compile(proposal.source,entry_point='metalmind_kernel'),
                                          fallback=fallback,domain_mode='checked')
        standard.proposal=proposal

        def fused(*inputs):
            if not matches_signature(workload,inputs):
                return fallback(*inputs)
            x,bias=inputs
            rejected=torch.zeros(1,dtype=torch.int32,device='mps')
            out=torch.empty_like(x)
            b,d=x.shape
            checked_kernel.function(x,bias,out,b,d,float(workload.epsilon),rejected,
                                    threads=proposal.threads,group_size=proposal.group_size)
            return fallback(*inputs) if rejected.item() else out

        fixtures=[tuple(t.to('mps') for t in f.inputs) for f in make_public_fixtures(workload)]
        for inputs in fixtures:
            assert compare_outputs(fused(*inputs),reference(*inputs),policy).passed
        assert fallback_count==0
        inputs=fixtures[0]
        cases=0
        for index,tensor in enumerate(inputs):
            for bits in (0,0x80000000,1,0x80000001,0x00800000,0x41200000,0x41200001,
                         0x7f800000,0x7fc00001,0xff800000):
                for position in (0,tensor.numel()//2,tensor.numel()-1):
                    cpu=tensor.cpu().clone()
                    cpu.view(torch.int32).flatten()[position]=bits if bits<2**31 else bits-2**32
                    updated=(*inputs[:index],cpu.to('mps'),*inputs[index+1:])
                    before=fallback_count
                    actual=fused(*updated)
                    expected=reference(*updated)
                    valid=inputs_in_domain(workload,updated)
                    assert fallback_count-before==int(not valid), (index,position,bits)
                    if valid:
                        assert compare_outputs(actual,expected,policy).passed
                    else:
                        torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
                    cases+=1
        baselines,unavailable=collect_baselines(workload)
        sessions=[]
        for seed in (202611501,202611502):
            measurements={'old_checked':pair(lambda:standard(*inputs),lambda:fused(*inputs),seed)}
            measurements.update({name:pair(lambda:fn(*inputs),lambda:fused(*inputs),seed) for name,fn in baselines})
            sessions.append(measurements)
    result={"artifact_id":manifest.artifact_id,"public_fixtures":len(fixtures),"boundary_cases":cases,
            "model_requests":0,"hardware":_hardware_profile(),"environment":asdict(probe_environment()),
            "unavailable":unavailable,"sessions":sessions}
    atomic_json(args.output,result)
    for session in sessions:
        print({name:{key:row[key] for key in ('status','candidate_ms','baseline_ms','summary','reason') if key in row}
               for name,row in session.items()},flush=True)

if __name__=='__main__':
    main()
