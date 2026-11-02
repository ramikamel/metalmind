"""Match exact reference expressions, keeping outside consumers intact."""
from dataclasses import dataclass
import builtins
import hashlib
import operator
import torch
import torch.nn.functional as F
from metalmind.graph.contracts import RegionInfo
from metalmind.graph.guards import tensor_spec
from metalmind.workloads.capture import normalize_graph
from metalmind.workloads.registry import _MODULES, workload_for_shape, get_workload

FUNCTIONS = {operator.add:"add",operator.mul:"mul",operator.truediv:"div",operator.pow:"pow",
    torch.add:"add",torch.mul:"mul",torch.rsqrt:"rsqrt",torch.softmax:"softmax",F.softmax:"softmax",F.gelu:"gelu"}

@dataclass
class CapturedRegion:
    info: RegionInfo
    output: torch.fx.Node
    nodes: set
    inputs: tuple
    example_inputs: tuple
    function: torch.fx.GraphModule


def _match(pattern, value, bindings, nodes):
    if isinstance(pattern, tuple) and pattern and pattern[0] == "input":
        if not isinstance(value, torch.fx.Node): return False
        index = pattern[1]
        if index in bindings: return bindings[index] is value
        bindings[index] = value
        return True
    if isinstance(pattern, tuple) and pattern and pattern[0] == "call":
        while isinstance(value, torch.fx.Node) and value.op == "call_method" and value.target in ("float", "to"):
            nodes.add(value); value = value.args[0]
        if not isinstance(value, torch.fx.Node): return False
        target = FUNCTIONS.get(value.target) if value.op == "call_function" else value.target if value.op == "call_method" else None
        if target != pattern[1]: return False
        kwargs = dict(value.kwargs);kwargs.pop("_stacklevel", None)
        nodes.add(value)
        return _match(pattern[2], value.args, bindings, nodes) and _match(pattern[3], tuple(sorted(kwargs.items())), bindings, nodes)
    if isinstance(pattern, tuple):
        return isinstance(value, (tuple, list)) and len(pattern) == len(value) and all(_match(p,v,bindings,nodes) for p,v in zip(pattern,value))
    return type(pattern) is type(value) and pattern == value


def _workload(family, inputs, output):
    specs = tuple(tensor_spec(t) for t in inputs)
    if any(s.device != "mps" for s in specs): return None, "Metal execution requires MPS; CPU metadata analysis only"
    if output.dtype != torch.float32 or output.device.type != "mps": return None, "Region output must be FP32 on MPS"
    spec = workload_for_shape(family, tuple(inputs[0].shape))
    if len(spec.inputs) != len(specs) or spec.inputs != specs or tensor_spec(output) != spec.output:
        return None, "Region metadata or broadcasting does not match the kernel contract"
    for case in ("small", "medium", "large"):
        known = get_workload(family, case)
        if known.inputs == spec.inputs: return known, None
    return spec, None


def partition_regions(graph, values):
    patterns = [(family, normalize_graph(module.reference)) for family,module in _MODULES.items()]
    regions = [];exclusions = [];claimed = set();order = list(graph.graph.nodes)
    for root in reversed(order):
        if root.op not in ("call_function", "call_method"): continue
        for family, pattern in patterns:
            bindings = {};nodes = set()
            if not _match(pattern, root, bindings, nodes): continue
            if nodes & claimed: continue
            if any(set(n.users) - nodes for n in nodes if n is not root):
                exclusions.append(root.name + ": supported expression has an outside consumer of an internal value")
                continue
            inputs = tuple(bindings[i] for i in range(len(bindings)))
            actual = tuple(values[n] for n in inputs)
            try:
                specs = tuple(tensor_spec(t) for t in actual)
                workload, reason = _workload(family, actual, values[root])
            except (ValueError, TypeError, AttributeError, IndexError) as exc:
                exclusions.append(root.name + ": " + str(exc));continue
            sub = torch.fx.Graph();mapping = {n:sub.placeholder("input_" + str(i)) for i,n in enumerate(inputs)}
            for n in order:
                if n in nodes:
                    def remap(other):
                        if other.op == "call_function" and other.target is builtins.getattr and other.args[1:] == ("dtype",):
                            return torch.float32
                        return mapping[other]
                    mapping[n] = sub.node_copy(n, remap)
            sub.output(mapping[root])
            function = torch.fx.GraphModule({}, sub)
            names = tuple(n.name for n in order if n in nodes)
            identity = hashlib.sha256((family + ":" + root.name + ":" + function.code).encode()).hexdigest()
            info = RegionInfo(region_id=identity,family=family,output_name=root.name,node_names=names,
                input_names=tuple(n.name for n in inputs),inputs=specs,workload=workload,
                estimated_input_bytes=sum(t.numel()*t.element_size() for t in actual),reason=reason)
            regions.append(CapturedRegion(info,root,nodes,inputs,actual,function));claimed.update(nodes)
            break
    regions.sort(key=lambda r:order.index(r.output))
    return regions, exclusions
