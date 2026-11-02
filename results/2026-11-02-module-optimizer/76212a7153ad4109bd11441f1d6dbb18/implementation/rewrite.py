"""Promoted local artifacts still need an independent whole-module gate."""
import copy
import hashlib
import json
from dataclasses import dataclass, field
import secrets
import time
from pathlib import Path
from typing import Literal
import uuid
import torch
from pydantic import Field
from metalmind.contracts import Contract
from metalmind.graph.capture import capture_module, RecordingInterpreter
from metalmind.graph.profile import analysis_from_capture, assert_measurement_ready
from metalmind.graph.contracts import ModuleAnalysis
from metalmind.artifacts.export import verify_artifact_evidence
from metalmind.artifacts.loader import load_artifact
from metalmind.artifacts.store import atomic_json
from metalmind.workloads.registry import resolve_reference
from metalmind.verification.numerics import inputs_in_domain
from metalmind.benchmarking.timer import measure_pair, BenchmarkUnstable
from metalmind.benchmarking.statistics import summarize_pairs
from metalmind.runtime.lock import gpu_lock
from metalmind.runtime.worker import _record_fault

class ModuleOptimizationConfig(Contract):
    artifact_paths: tuple[Path, ...] = ()
    domain_mode: Literal["checked", "assume"] = "checked"
    output_dir: Path | None = None
    private_cases: int = Field(default=8,ge=8,le=32)
    confirmation_pairs: int = Field(default=30,ge=30)
    bootstrap_resamples: int = Field(default=10000,ge=1000)
    confirmation_cooldown_seconds: float = Field(default=60,ge=60)
    wall_seconds: float = Field(default=600,gt=60)

@dataclass
class OptimizedModuleResult:
    module: object
    enabled: bool
    outcome: str
    reason: str
    analysis: ModuleAnalysis
    run_path: Path | None = None
    artifact_ids: tuple[str, ...] = ()
    private_verification: dict | None = None
    confirmations: list = field(default_factory=list)
    candidate_id: str | None = None

    def report(self):
        return {"enabled":self.enabled,"outcome":self.outcome,"reason":self.reason,
            "analysis":self.analysis.model_dump(mode="json"),"run_path":str(self.run_path) if self.run_path else None,
            "artifact_ids":self.artifact_ids,"private_verification":self.private_verification,
            "confirmations":self.confirmations,"candidate_id":self.candidate_id,"model_requests":0,
            "baseline":"original eager whole module; no compiled whole-module comparison"}

class GuardedModule:
    """A callable wrapper avoids adding extra nn.Module hooks on fallback calls."""
    def __init__(self, original, rewritten, guard):
        self.original = original;self._rewritten = rewritten;self._guard = guard
        self.last_path = None
    def __call__(self, *inputs, **kwargs):
        if not kwargs and self._guard.matches(inputs):
            self.last_path = "optimized"
            return self._rewritten(*inputs)
        self.last_path = "fallback"
        return self.original(*inputs,**kwargs)

class _ArtifactRegion(torch.nn.Module):
    def __init__(self, function):
        super().__init__();self.function = function
    def forward(self, *inputs):
        return self.function(*inputs)


def _rewrite_graph(captured, replacements):
    graph = copy.deepcopy(captured.graph.graph)
    rewritten = torch.fx.GraphModule(captured.graph,graph)
    nodes = {n.name:n for n in graph.nodes}
    for index, region in enumerate(captured.regions):
        function = replacements.get(region.info.region_id)
        if function is None: continue
        root = nodes[region.output.name]
        name = "_metalmind_region_" + str(index)
        rewritten.add_submodule(name,_ArtifactRegion(function).eval())
        inputs = tuple(nodes[n.name] for n in region.inputs)
        with graph.inserting_before(root):
            replacement = graph.call_module(name,inputs)
        root.replace_all_uses_with(replacement)
        nodes[region.output.name] = replacement
        for node in reversed(list(graph.nodes)):
            if node.name in region.info.node_names and not node.users:
                graph.erase_node(node)
    graph.lint();rewritten.recompile()
    return rewritten


def _semantic_key(workload):
    return workload.model_dump(mode="json",exclude={"workload_id"})


def _select_artifacts(captured, config):
    selected = {};identities = [];issues = []
    for path in config.artifact_paths:
        try:
            manifest = verify_artifact_evidence(path)
            if manifest.domain_mode != config.domain_mode:
                raise ValueError("Artifact domain mode differs; assumed-domain reuse requires explicit opt-in")
            if manifest.workload.reference_id != manifest.workload.family:
                raise ValueError("Module extraction requires a registered semantic reference artifact")
            matches = [r for r in captured.regions if r.info.workload and
                _semantic_key(r.info.workload) == _semantic_key(manifest.workload) and r.info.region_id not in selected]
            if not matches: continue
            fn = load_artifact(path,fallback=resolve_reference(manifest.workload.family))
            if fn.optimized is None: raise ValueError(fn.reason)
            for region in matches: selected[region.info.region_id] = fn
            identities.append(manifest.artifact_id)
        except (ValueError,OSError,RuntimeError,TypeError,KeyError) as exc:
            issues.append(str(path) + ": " + str(exc)[:400])
    return selected,tuple(identities),issues


def _compare(actual, expected):
    if isinstance(expected,torch.Tensor):
        if not isinstance(actual,torch.Tensor) or not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
            raise ValueError("Whole-module outputs must be finite matching tensors")
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4,equal_nan=False)
        return (actual-expected).abs().max().item() if actual.numel() else 0.
    if type(actual) is not type(expected): raise ValueError("Output structure changed")
    if isinstance(expected,dict):
        if list(actual) != list(expected): raise ValueError("Output keys changed")
        return max((_compare(actual[k],v) for k,v in expected.items()),default=0.)
    if isinstance(expected,(tuple,list)):
        if len(actual) != len(expected): raise ValueError("Output length changed")
        return max((_compare(a,b) for a,b in zip(actual,expected)),default=0.)
    if actual != expected: raise ValueError("Non-tensor output changed")
    return 0.


def _versions(tensors):
    values = []
    for t in tensors:
        try: values.append(t._version)
        except RuntimeError: values.append(None)
    return tuple(values)


def _verify_private(captured, candidate, selected, config, seeds):
    maximum = 0.
    parameters = tuple(captured.original.parameters()) + tuple(captured.original.buffers())
    for seed in seeds:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        inputs = tuple((torch.randn(tuple(t.shape),generator=rng)*.25).clamp_(-1,1).to(t.device) for t in captured.example_inputs)
        before = _versions((*inputs,*parameters))
        with torch.inference_mode():
            original_graph = RecordingInterpreter(captured.graph);original_graph.run(*inputs)
            for region in captured.regions:
                if region.info.region_id in selected:
                    values = tuple(original_graph.values[n] for n in region.inputs)
                    if not inputs_in_domain(region.info.workload,values):
                        raise ValueError("Private case leaves an optimized region's declared numerical domain")
            expected = captured.original(*inputs);actual = candidate(*inputs)
            if candidate.last_path != "optimized": raise ValueError("Module guard changed during private verification")
            maximum = max(maximum,_compare(actual,expected))
        if before != _versions((*inputs,*parameters)):
            raise ValueError("Module inputs or parameters mutated during private verification")
    return {"passed":True,"cases":len(seeds),"atol":1e-5,"rtol":1e-4,
        "max_absolute_error":maximum,"input_distribution":"CPU seeded normal*0.25, clipped to [-1,1]",
        "parameters":"live current values; independent parameter-value testing belongs to the artifact gate"}


def _confirm(original, candidate, config, *, save=None, deadline=None):
    confirmations = []
    for session in range(2):
        if session:
            if deadline is not None and time.monotonic()+config.confirmation_cooldown_seconds >= deadline:
                return "budget_exhausted",confirmations
            time.sleep(config.confirmation_cooldown_seconds)
        if deadline is not None and time.monotonic() >= deadline: return "budget_exhausted",confirmations
        try:
            series = measure_pair(original,candidate,pairs=config.confirmation_pairs,seed=71+session)
        except BenchmarkUnstable as exc:
            confirmations.append({"session":session+1,"status":"inconclusive","reason":str(exc)})
            if save: save(session+1,confirmations[-1])
            return "inconclusive",confirmations
        summary = summarize_pairs(series.baseline_ms,series.candidate_ms,seed=71+session,resamples=config.bootstrap_resamples)
        evidence = {"session":session+1,"status":"valid","baseline":"original_eager_whole_module",
            "boundary":"guarded whole module, including all unchanged operations",
            "domain_mode":config.domain_mode,"seed":71+session,"bootstrap_resamples":config.bootstrap_resamples,
            "series":series.model_dump(mode="json"),"summary":summary.model_dump(mode="json")}
        confirmations.append(evidence)
        if save: save(session+1,evidence)
        if deadline is not None and time.monotonic() >= deadline: return "budget_exhausted",confirmations
        if summary.speedup < 1.05 or summary.lower <= 1.: return "no_win",confirmations
    return "promoted",confirmations


def optimize_module(module, example_inputs, *, config=None):
    config = config or ModuleOptimizationConfig()
    run = None;analysis = ModuleAnalysis(supported=False,reason="Capture has not completed")
    if config.output_dir is not None:
        run = config.output_dir / uuid.uuid4().hex;run.mkdir(parents=True,exist_ok=False)
        atomic_json(run/"manifest.json",{"phase":"capturing","config":config.model_dump(mode="json"),"model_requests":0})
    private = None;confirmations = [];identities = ();candidate = None;candidate_id = None
    started = time.monotonic()
    def finish(outcome, reason):
        result = OptimizedModuleResult(candidate if outcome=="promoted" else module,outcome=="promoted",
            outcome,reason,analysis,run,identities,private,confirmations,candidate_id)
        if run:
            atomic_json(run/"result.json",result.report())
            manifest = {"phase":"terminal","config":config.model_dump(mode="json"),"analysis":analysis.model_dump(mode="json"),"model_requests":0,"artifact_ids":identities,"candidate_id":candidate_id,"elapsed_seconds":time.monotonic()-started}
            atomic_json(run/"manifest.json",manifest)
        return result
    try:
        captured = capture_module(module,example_inputs)
        analysis = analysis_from_capture(captured)
        if run: (run/"original-graph.py").write_text(captured.graph.code)
        if not config.artifact_paths or not any(r.info.workload for r in captured.regions):
            return finish("no_artifact","No matching promoted MPS artifact selected; original module retained")
        assert_measurement_ready()
        with gpu_lock():
            selected,identities,issues = _select_artifacts(captured,config)
            analysis = analysis.model_copy(update={"exclusions":analysis.exclusions+tuple(issues)})
            if not selected: return finish("no_artifact","No compatible artifact matched the region signature and explicitly selected domain mode")
            for region in captured.regions:
                if region.info.region_id in selected and not inputs_in_domain(region.info.workload,region.example_inputs):
                    raise ValueError("Example inputs leave an optimized region's declared numerical domain")
            rewritten = _rewrite_graph(captured,selected)
            candidate = GuardedModule(module,rewritten,captured.guard)
            implementation = {p.name:p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))}
            identity = {"original_graph":captured.graph_hash,"artifact_ids":identities,"domain_mode":config.domain_mode,
                "rewritten_graph":rewritten.code,"implementation":implementation,"guard_policy_version":1}
            candidate_id = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            if run:
                snapshots = run/"implementation";snapshots.mkdir()
                for name,source in implementation.items(): (snapshots/name).write_text(source)
                atomic_json(run/"candidate-identity.json",{k:v for k,v in identity.items() if k!="implementation"})
                (run/"candidate-graph.py").write_text(rewritten.code)
                atomic_json(run/"selected-artifacts.json",{"artifact_ids":identities,"regions":tuple(selected)})
            seeds = tuple(secrets.randbits(63) for _ in range(config.private_cases))
            if run: atomic_json(run/"private-fixtures.json",{"seeds":seeds,"cases":config.private_cases})
            private = _verify_private(captured,candidate,selected,config,seeds)
            if run: atomic_json(run/"private-verification.json",private)
            def save(session,evidence):
                if run: atomic_json(run/f"confirmation-{session}.json",evidence)
            outcome,confirmations = _confirm(lambda:module(*captured.example_inputs),lambda:candidate(*captured.example_inputs),
                config,save=save,deadline=started+config.wall_seconds)
            reasons = {"promoted":"Private whole-module outputs and both integrated timing sessions passed",
                "no_win":"Local kernels did not establish the required whole-module gain; original module retained",
                "inconclusive":"An integrated timing window was unstable; original module retained",
                "budget_exhausted":"Integrated verification exceeded its budget; original module retained"}
            return finish(outcome,reasons[outcome])
    except (ValueError,TypeError,OSError,AssertionError,AttributeError) as exc:
        if candidate is not None and private is None:
            private = {"passed":False,"reason":str(exc)[:1000]}
            if run: atomic_json(run/"private-verification.json",private)
        return finish("unsupported" if not analysis.supported else "failed",str(exc)[:1000])
    except RuntimeError as exc:
        if not any(isinstance(t,torch.Tensor) and t.device.type == "mps" for t in example_inputs):
            return finish("unsupported",str(exc)[:1000])
        _record_fault(str(exc))
        return finish("failed","GPU/runtime failure; health validation required: " + str(exc)[:500])
