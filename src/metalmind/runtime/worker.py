"""Spawned local evaluator. Fault containment, not a hostile-code sandbox."""
import multiprocessing as mp
import os
import time
from pathlib import Path
import tempfile

FAULT_PATH=Path(tempfile.gettempdir())/f"metalmind-gpu-{os.getuid()}-fault.json"

def gpu_faulted(): return FAULT_PATH.exists()

def clear_device_fault(): FAULT_PATH.unlink(missing_ok=True)

def _record_fault(reason):
    from metalmind.artifacts.store import atomic_json
    atomic_json(FAULT_PATH,{"reason":reason,"recovery":"Run metalmind doctor --smoke successfully before further GPU evaluation"})
from metalmind.contracts import EvaluationRequest, EvaluationResult


def _child(raw,connection):
    os.environ.pop("MODEL_API_KEY",None)
    try:
        from metalmind.runtime.metal import MetalBackend, ShaderCompilationError
        from metalmind.runtime.abi import signature_for
        from metalmind.workloads.registry import fingerprint_reference
        from metalmind.workloads.fixtures import make_public_fixtures,make_private_fixtures
        from metalmind.verification.numerics import verify_candidate
        request=EvaluationRequest.model_validate(raw)
        if request.candidate.workload_hash!=signature_for(request.workload):
            raise ValueError("workload signature mismatch")
        if fingerprint_reference(request.workload.reference_id)!=request.workload.reference_hash:
            raise ValueError("reference changed")
        if request.workload.reference_id.startswith('python:'):
            from metalmind.workloads.registry import resolve_reference,_MODULES
            from metalmind.workloads.capture import _validate_source_before_trace,normalize_graph
            original=resolve_reference(request.workload.reference_id)
            _validate_source_before_trace(original)
            if normalize_graph(original)!=normalize_graph(_MODULES[request.workload.family].reference):
                raise ValueError('user reference no longer matches supported semantics')
        from metalmind.agent.candidates import validate_proposal
        validate_proposal(request.candidate.proposal.model_dump_json())
        source=request.candidate.proposal.source
        if len(source.encode())>65536: raise ValueError("source exceeds byte budget")
        import re
        includes=re.findall(r'\s*#\s*include\s*[<"]([^>"]+)',source,re.MULTILINE)
        if any(name!="metal_stdlib" for name in includes): raise ValueError("unapproved include")
        backend=MetalBackend()
        try: kernel=backend.compile(source,entry_point=request.candidate.proposal.entry_point)
        except ShaderCompilationError as exc:
            connection.send({"result":EvaluationResult(status="compile_error",reason=str(exc)[:4096]).model_dump()});return
        if __import__('math').prod(request.candidate.proposal.group_size)>kernel.max_threads_per_threadgroup:
            raise ValueError("threadgroup exceeds pipeline limit")
        connection.send({"stage":"execution"})
        import torch
        from math import prod
        item_size=2 if request.workload.inputs[0].dtype=="float16" else 4
        fixture_count=22 if request.mode=="final_verify" else 10
        planned=sum(prod(s.shape)*item_size for s in request.workload.inputs)*(fixture_count*2+16)+prod(request.workload.output.shape)*item_size*8
        if planned>torch.mps.recommended_max_memory()*.2:
            raise ValueError("planned live fixture/scratch memory exceeds 20% of recommended working set")
        fixtures=make_private_fixtures(request.workload,request.private_seeds) if request.mode=="final_verify" else make_public_fixtures(request.workload)
        verification=verify_candidate(request.candidate,request.workload,fixtures,backend=backend,kernel=kernel)
        result=EvaluationResult(status="valid" if verification.passed else "incorrect",verification=verification)
        if verification.passed and request.measure:
            from metalmind.benchmarking.timer import evaluate_performance
            result=evaluate_performance(request,backend,kernel,verification)
        connection.send({"result":result.model_dump()})
    except Exception as exc:
        connection.send({"result":EvaluationResult(status="runtime_error",reason=f"{type(exc).__name__}: {exc}"[:4096],device_fault=isinstance(exc,RuntimeError) and type(exc).__name__!="BenchmarkUnstable").model_dump()})
    finally: connection.close()


def evaluate(request,cancel_event=None):
    from metalmind.runtime.lock import gpu_lock
    if cancel_event is not None and cancel_event.is_set(): return EvaluationResult(status="cancelled")
    if gpu_faulted(): return EvaluationResult(status="runtime_error",device_fault=True,reason="GPU fault latch is set; run doctor --smoke to recover")
    started=time.monotonic()
    absolute=started+(request.total_timeout_seconds or request.compile_timeout_seconds+request.execution_timeout_seconds)
    ctx=mp.get_context("spawn");receive,send=ctx.Pipe(duplex=False)
    process=ctx.Process(target=_child,args=(request.model_dump(),send),daemon=True)
    def fault(status,reason):
        _record_fault(reason)
        return EvaluationResult(status=status,device_fault=True,reason=reason)
    try:
        with gpu_lock(min(request.compile_timeout_seconds,max(.001,absolute-time.monotonic()))):
            if time.monotonic()>=absolute: return EvaluationResult(status="timeout",reason="total deadline expired during lock acquisition")
            process.start();send.close()
            deadline=min(absolute,time.monotonic()+request.compile_timeout_seconds)
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return fault("cancelled","evaluation interrupted; health validation required")
                if time.monotonic()>=deadline: return fault("timeout","worker absolute/stage deadline exceeded; health validation required")
                if receive.poll(min(.05,max(.001,deadline-time.monotonic()))):
                    try: message=receive.recv()
                    except EOFError: return fault("runtime_error","worker exited without a result")
                    if "result" in message:
                        if time.monotonic()>=absolute: return fault("timeout","result arrived after absolute deadline")
                        result=EvaluationResult.model_validate(message["result"])
                        if result.device_fault: _record_fault(result.reason)
                        return result
                    if message.get("stage")=="execution": deadline=min(absolute,time.monotonic()+request.execution_timeout_seconds)
                if not process.is_alive() and not receive.poll(): return fault("runtime_error","worker crashed")
    except TimeoutError as exc: return EvaluationResult(status="timeout",reason=str(exc))
    finally:
        if process.pid is not None:
            if process.is_alive(): process.terminate()
            process.join(timeout=2)
            if process.is_alive(): process.kill();process.join(timeout=2)
        receive.close();send.close()
