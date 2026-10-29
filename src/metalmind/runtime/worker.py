"""Spawned local evaluator. Fault containment, not a hostile-code sandbox."""
import multiprocessing as mp
import os
import time
from metalmind.contracts import EvaluationRequest, EvaluationResult


def _child(raw,connection):
    os.environ.pop("MODEL_API_KEY",None)
    try:
        from metalmind.runtime.metal import MetalBackend, ShaderCompilationError
        from metalmind.runtime.abi import signature_for
        from metalmind.workloads.registry import reference_hash
        from metalmind.workloads.fixtures import make_public_fixtures,make_private_fixtures
        from metalmind.verification.numerics import verify_candidate
        request=EvaluationRequest.model_validate(raw)
        if request.candidate.workload_hash!=signature_for(request.workload):
            raise ValueError("workload signature mismatch")
        if reference_hash(request.workload.family)!=request.workload.reference_hash:
            raise ValueError("reference changed")
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
        fixtures=make_private_fixtures(request.workload,request.private_seeds) if request.mode=="final_verify" else make_public_fixtures(request.workload)
        verification=verify_candidate(request.candidate,request.workload,fixtures,backend=backend,kernel=kernel)
        result=EvaluationResult(status="valid" if verification.passed else "incorrect",verification=verification)
        if verification.passed and request.measure:
            from metalmind.benchmarking.timer import evaluate_performance
            result=evaluate_performance(request,backend,kernel,verification)
        connection.send({"result":result.model_dump()})
    except Exception as exc:
        connection.send({"result":EvaluationResult(status="runtime_error",reason=f"{type(exc).__name__}: {exc}"[:4096]).model_dump()})
    finally: connection.close()


def evaluate(request,cancel_event=None):
    from metalmind.runtime.lock import gpu_lock
    if cancel_event is not None and cancel_event.is_set(): return EvaluationResult(status="cancelled")
    ctx=mp.get_context("spawn");receive,send=ctx.Pipe(duplex=False)
    process=ctx.Process(target=_child,args=(request.model_dump(),send),daemon=True)
    try:
        with gpu_lock(request.compile_timeout_seconds):
            process.start();send.close()
            deadline=time.monotonic()+request.compile_timeout_seconds
            while True:
                if cancel_event is not None and cancel_event.is_set(): return EvaluationResult(status="cancelled")
                if time.monotonic()>deadline: return EvaluationResult(status="timeout",reason="worker stage deadline exceeded; stop further GPU evaluation until healthy")
                if receive.poll(.05):
                    try: message=receive.recv()
                    except EOFError: return EvaluationResult(status="runtime_error",reason="worker exited without a result")
                    if "result" in message: return EvaluationResult.model_validate(message["result"])
                    if message.get("stage")=="execution": deadline=time.monotonic()+request.execution_timeout_seconds
                if not process.is_alive() and not receive.poll(): return EvaluationResult(status="runtime_error",reason="worker crashed")
    except TimeoutError as exc: return EvaluationResult(status="timeout",reason=str(exc))
    finally:
        if process.pid is not None:
            if process.is_alive(): process.terminate()
            process.join(timeout=2)
            if process.is_alive(): process.kill();process.join(timeout=2)
        receive.close();send.close()
