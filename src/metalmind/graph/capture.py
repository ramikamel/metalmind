"""Admit trusted flat pure modules before executing source or FX tracing."""
from dataclasses import dataclass
import ast
import builtins
import hashlib
import inspect
import json
import operator
import textwrap
import types
import torch
import torch.nn.functional as F
from metalmind.graph.guards import ModuleGuard, tensor_spec, forward_function, hooks_present
from metalmind.graph.regions import partition_regions

PURE_NAMES = {"gelu","softmax","rsqrt","square","mean","pow","add","mul","div","sub","sum","sqrt",
    "sin","cos","matmul","mm","linear","transpose","t","contiguous","float","to","sigmoid","tanh","relu"}
PURE_FUNCTIONS = {getattr(torch,n) for n in PURE_NAMES if hasattr(torch,n) and callable(getattr(torch,n))} | {F.gelu,F.softmax,F.linear,F.relu}
PURE_FUNCTIONS |= {operator.add,operator.sub,operator.mul,operator.truediv,operator.pow,operator.matmul,operator.getitem}

@dataclass
class ModuleCapture:
    original: torch.nn.Module
    graph: torch.fx.GraphModule
    regions: list
    values: dict
    exclusions: list
    guard: ModuleGuard
    graph_hash: str
    example_inputs: tuple

class RecordingInterpreter(torch.fx.Interpreter):
    def __init__(self, module):
        super().__init__(module);self.values = {}
    def run_node(self, node):
        result = super().run_node(node);self.values[node] = result;return result


def _admit(module, inputs):
    if not isinstance(module, torch.nn.Module) or module.training or list(module.children()) or hooks_present(module):
        raise ValueError("Module must be a flat eval-mode module without hooks")
    if (type(module).__call__ is not torch.nn.Module.__call__ or type(module).__getattribute__ is not torch.nn.Module.__getattribute__
        or type(module).__getattr__ is not torch.nn.Module.__getattr__ or type(module)._call_impl is not torch.nn.Module._call_impl):
        raise ValueError("Custom module invocation/attribute behavior is unsupported")
    if not inputs: raise ValueError("At least one positional tensor input is required")
    for t in (*inputs, *module.parameters(), *module.buffers()): tensor_spec(t)
    fn = forward_function(module);source = textwrap.dedent(inspect.getsource(fn));tree = ast.parse(source)
    forbidden = (ast.If,ast.For,ast.While,ast.With,ast.Try,ast.Raise,ast.Await,ast.Yield,ast.Lambda,
        ast.AsyncFunctionDef,ast.Global,ast.Nonlocal,ast.AugAssign,ast.Delete,ast.Import,ast.ImportFrom,ast.NamedExpr,ast.IfExp,ast.BoolOp)
    scalar_names = set();callees = set();methods = set();attributes = set()
    local_names = set(fn.__code__.co_varnames)
    operators = {ast.Add:"__add__",ast.Sub:"__sub__",ast.Mult:"__mul__",ast.Div:"__truediv__",ast.Pow:"__pow__",ast.MatMult:"__matmul__",ast.USub:"__neg__"}
    tensor_names = set(dict(module.named_parameters())) | set(dict(module.named_buffers()))
    for node in ast.walk(tree):
        if type(node) in operators: methods.add(operators[type(node)])
        if isinstance(node,ast.Subscript): methods.add("__getitem__")
        if isinstance(node, forbidden): raise ValueError("Control flow, mutation and Python side effects are unsupported")
        if isinstance(node, ast.Expr) and not (isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)):
            raise ValueError("Ignored expressions may have side effects")
        if isinstance(node,(ast.Attribute,ast.Subscript)) and isinstance(node.ctx,ast.Store):
            raise ValueError("Mutation is unsupported")
        if isinstance(node, ast.Assign) and any(not isinstance(t,ast.Name) for t in node.targets):
            raise ValueError("Mutation and complex assignments are unsupported")
        if isinstance(node, ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id == "self":
            if node.attr not in tensor_names:
                value = vars(module).get(node.attr)
                if node.attr not in vars(module) or type(value) not in (str,int,float,bool,type(None)):
                    raise ValueError("Only registered tensors and direct scalar module attributes are supported")
                scalar_names.add(node.attr)
        if isinstance(node,ast.Attribute):
            path = [];root = node
            while isinstance(root,ast.Attribute): path.insert(0,root.attr);root=root.value
            if isinstance(root,ast.Name) and root.id not in local_names and isinstance(fn.__globals__.get(root.id),types.ModuleType):
                attributes.add((root.id,tuple(path)))
        if isinstance(node,ast.Assign):
            root = node.value;path = []
            while isinstance(root,ast.Attribute): path.insert(0,root.attr);root=root.value
            if isinstance(root,ast.Name) and root.id not in local_names:
                value = fn.__globals__.get(root.id)
                if isinstance(value,types.ModuleType):
                    for name in path: value=getattr(value,name)
                    if isinstance(value,types.ModuleType): raise ValueError("Local aliases of Python modules are unsupported")
        if isinstance(node, ast.Call):
            if any(k.arg in (None,"out","inplace") for k in node.keywords):
                raise ValueError("Output mutation and dynamic keyword arguments are unsupported")
            if not isinstance(node.func,ast.Attribute) or node.func.attr not in PURE_NAMES:
                raise ValueError("Unsupported call before module tracing")
            path = [];value = node.func
            while isinstance(value,ast.Attribute): path.insert(0,value.attr);value=value.value
            if isinstance(value,ast.Name) and value.id not in local_names and value.id in fn.__globals__:
                target = fn.__globals__[value.id]
                if target not in (torch,F): raise ValueError("Unsupported global callable")
                for name in path: target = getattr(target,name)
                if target not in PURE_FUNCTIONS: raise ValueError("Unsupported global callable")
                callees.add((value.id,tuple(path)))
            else: methods.add(node.func.attr)
    guard = ModuleGuard(module, inputs, scalar_names=scalar_names, callees=callees, methods=methods, attributes=attributes)
    return guard, source


def _admit_graph(graph):
    for n in graph.graph.nodes:
        if any(k in n.kwargs for k in ("out","inplace")):
            raise ValueError("Output mutation is unsupported")
        if n.op in ("placeholder","get_attr","output"): continue
        if n.op == "call_function" and (n.target in PURE_FUNCTIONS or (n.target is builtins.getattr and n.args[1:] == ("dtype",))): continue
        if n.op == "call_method" and n.target in PURE_NAMES:
            if n.target == "float" and (len(n.args) != 1 or n.kwargs): raise ValueError("Unsupported float conversion")
            if n.target == "to":
                if len(n.args)>2 or set(n.kwargs)-{"dtype"} or (len(n.args)==2 and n.kwargs): raise ValueError("Device transfers and additional cast options are unsupported")
                dtype = n.args[1] if len(n.args)==2 else n.kwargs.get("dtype")
                if dtype is not torch.float32 and not (isinstance(dtype,torch.fx.Node) and dtype.op=="call_function" and dtype.target is builtins.getattr and dtype.args[1:]==("dtype",)):
                    raise ValueError("Only FP32-preserving conversions are supported")
            continue
        raise ValueError("Unsupported or stateful FX operation: " + str(n.target))


def capture_module(module, example_inputs):
    inputs = tuple(example_inputs);guard,source = _admit(module,inputs)
    graph = torch.fx.symbolic_trace(module);_admit_graph(graph)
    from contextlib import nullcontext
    from metalmind.runtime.lock import gpu_lock
    from metalmind.runtime.worker import gpu_faulted, _record_fault
    is_mps = any(t.device.type == "mps" for t in inputs)
    if is_mps and gpu_faulted(): raise ValueError("GPU fault latch is set; run doctor --smoke")
    with gpu_lock() if is_mps else nullcontext(), torch.inference_mode():
        try:
            interpreter = RecordingInterpreter(graph);actual = interpreter.run(*inputs)
            torch.testing.assert_close(actual,module(*inputs),atol=0,rtol=0)
        except RuntimeError as exc:
            if is_mps: _record_fault("Module capture runtime failure: " + str(exc)[:500])
            raise
    if any(isinstance(v,torch.Tensor) and v.dtype != torch.float32 for v in interpreter.values.values()):
        raise ValueError("Initial module capture requires FP32 intermediates")
    regions,exclusions = partition_regions(graph,interpreter.values)
    identity = {"source":source,"graph":graph.code,"inputs":[s.model_dump(mode="json") for s in guard.input_specs],
        "state":repr(guard.state[1]),"parameters":[(n,tensor_spec(t).model_dump(mode="json")) for n,t in (*module.named_parameters(),*module.named_buffers())]}
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return ModuleCapture(module,graph,regions,interpreter.values,exclusions,guard,fingerprint,inputs)
