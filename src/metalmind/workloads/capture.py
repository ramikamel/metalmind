"""Restricted exact FX pattern recognition for trusted importable functions."""
import builtins
import ast
import textwrap
import dis
import importlib
import inspect
import math
import operator
from pathlib import Path
import types
import torch
import torch.nn.functional as F
from metalmind.contracts import Contract
from metalmind.workloads.registry import workload_for_shape,_MODULES

class UnsupportedReason(Contract):
    reason: str

def describe_reference(fn):
    if not inspect.isfunction(fn) or fn.__closure__ or '<' in fn.__qualname__ or '.' in fn.__qualname__:
        raise ValueError("Only importable top-level functions without closures are supported")
    module=importlib.import_module(fn.__module__)
    if getattr(module,fn.__qualname__,None) is not fn: raise ValueError("function is not importable by identity")
    for name in {i.argval for i in dis.get_instructions(fn) if i.opname=='LOAD_GLOBAL'}:
        value=fn.__globals__.get(name)
        if isinstance(value,types.ModuleType):
            if value.__name__ not in ('torch','torch.nn.functional','math'): raise ValueError("unsupported global module")
        elif value is not None and not isinstance(value,(int,float,str,bool)) and value not in (F.gelu,torch.softmax,torch.rsqrt):
            raise ValueError("mutable or unsupported global capture")
    if any(not isinstance(v,(int,float,str,bool,type(None))) for v in (fn.__defaults__ or ())):
        raise ValueError("mutable function defaults")
    return f"python:{fn.__module__}:{fn.__qualname__}"


def _validate_source_before_trace(fn):
    tree=ast.parse(textwrap.dedent(inspect.getsource(fn)))
    forbidden=(ast.If,ast.For,ast.While,ast.With,ast.Try,ast.Raise,ast.Await,ast.Yield,ast.Lambda,
        ast.AsyncFunctionDef,ast.Global,ast.Nonlocal,ast.AugAssign,ast.Delete,ast.Import,ast.ImportFrom)
    methods={'float','to','square','mean','pow','softmax','gelu','rsqrt','add','mul','div','sqrt'}
    for node in ast.walk(tree):
        if isinstance(node,forbidden): raise ValueError('control flow, mutation, and Python side effects are unsupported')
        if isinstance(node,ast.Expr) and not (isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)):
            raise ValueError('ignored expressions may have side effects')
        if isinstance(node,ast.Assign) and any(not isinstance(t,ast.Name) for t in node.targets):
            raise ValueError('mutation or complex assignment is unsupported')
        if isinstance(node,ast.Call):
            if isinstance(node.func,ast.Attribute):
                if node.func.attr not in methods: raise ValueError('unsupported call before tracing: '+node.func.attr)
            elif isinstance(node.func,ast.Name):
                if fn.__globals__.get(node.func.id) not in (F.gelu,torch.softmax,torch.rsqrt):
                    raise ValueError('unsupported Python call before tracing')
            else: raise ValueError('dynamic callable is unsupported')


def normalize_graph(fn):
    graph=torch.fx.symbolic_trace(fn).graph
    values={};count=0
    functions={operator.add:'add',operator.mul:'mul',operator.truediv:'div',operator.pow:'pow',
        torch.add:'add',torch.mul:'mul',torch.rsqrt:'rsqrt',torch.softmax:'softmax',F.softmax:'softmax',F.gelu:'gelu'}
    def value(x):
        if isinstance(x,torch.fx.Node): return values[x]
        if isinstance(x,(tuple,list)): return tuple(value(v) for v in x)
        if isinstance(x,dict): return tuple(sorted((k,value(v)) for k,v in x.items()))
        if isinstance(x,torch.dtype): return str(x)
        if isinstance(x,(int,float,str,bool,type(None))): return x
        raise ValueError("unsupported graph constant")
    for node in graph.nodes:
        if node.op=='placeholder': values[node]=('input',count);count+=1
        elif node.op=='call_function' and node.target is builtins.getattr:
            if node.args[1]!='dtype': raise ValueError("unsupported tensor attribute")
            values[node]=('dtype',)
        elif node.op=='call_method' and node.target in ('float','to'):
            if node.target=='float':
                if len(node.args)!=1 or node.kwargs: raise ValueError("unsupported cast arguments")
            else:
                if len(node.args)>2 or set(node.kwargs)-{'dtype'} or (len(node.args)==2 and node.kwargs):
                    raise ValueError("device transfers and additional cast options are unsupported")
                dtype=value(node.args[1]) if len(node.args)>1 else value(node.kwargs.get('dtype'))
                if dtype not in (('dtype',),'torch.float32'): raise ValueError("unsupported dtype/device conversion")
            values[node]=value(node.args[0])
        elif node.op in ('call_function','call_method'):
            target=functions.get(node.target) if node.op=='call_function' else node.target
            if target not in ('add','mul','div','pow','rsqrt','softmax','gelu','square','mean'):
                raise ValueError(f"unsupported graph operation: {node.target}")
            kwargs=dict(node.kwargs);kwargs.pop('_stacklevel',None)
            values[node]=('call',target,value(node.args),value(kwargs))
        elif node.op=='output': return value(node.args[0])
        else: raise ValueError("state, mutation, or unsupported graph node")
    raise ValueError("no output")


def capture_function(fn,example_inputs):
    try:
        reference_id=describe_reference(fn)
        _validate_source_before_trace(fn)
        if not example_inputs or any(not isinstance(t,torch.Tensor) or t.dtype!=torch.float32 or not t.is_contiguous() or t.storage_offset()!=0 for t in example_inputs):
            raise ValueError("Initial public capture requires contiguous zero-offset FP32 tensors")
        graph=normalize_graph(fn)
        for family,module in _MODULES.items():
            if graph==normalize_graph(module.reference):
                spec=workload_for_shape(family,tuple(example_inputs[0].shape))
                if len(example_inputs)!=len(spec.inputs) or any(tuple(t.shape)!=s.shape for t,s in zip(example_inputs,spec.inputs)):
                    raise ValueError("input metadata disagrees with supported broadcasting")
                from metalmind.workloads.registry import fingerprint_reference
                return spec.model_copy(update={"reference_id":reference_id,"reference_hash":fingerprint_reference(reference_id)})
        raise ValueError("Function does not match a supported exact semantic pattern")
    except (ValueError,TypeError,RuntimeError,OSError,NameError,torch.fx.proxy.TraceError) as exc:
        return UnsupportedReason(reason=str(exc))
