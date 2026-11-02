"""Cheap runtime guards; tensor values remain live and are never frozen."""
import ast
import dis
import inspect
import os
import textwrap
import types
import torch
from metalmind.contracts import TensorSpec

HOOKS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
FLAGS = ("PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_MPS_METAL_VERSION", "PYTORCH_ENABLE_MPS_FALLBACK", "MTL_SHADER_VALIDATION")


def tensor_key(t):
    if type(t) not in (torch.Tensor,torch.nn.Parameter) or t.dtype != torch.float32:
        raise ValueError("Module capture requires ordinary FP32 tensors")
    device = t.device
    shape = tuple(t.shape);strides = tuple(t.stride());offset = t.storage_offset()
    if not t.is_contiguous() or offset != 0 or not shape:
        raise ValueError("Module capture requires contiguous zero-offset nonscalar tensors")
    if device.type not in ("mps","cpu"):
        raise ValueError("Module capture supports CPU metadata or MPS execution")
    return shape,strides,device.type,device.index,offset


def tensor_spec(t):
    shape,strides,device,_,offset = tensor_key(t)
    return TensorSpec(shape=shape,strides=strides,dtype="float32",device=device,storage_offset=offset)


def _registered_tensors(module):
    return tuple((name,t) for mapping in (module._parameters,module._buffers) for name,t in mapping.items() if t is not None)


def _descriptor(cls, name):
    for base in cls.__mro__:
        if name in base.__dict__: return id(base.__dict__[name])
    return id(None)


def hooks_present(module):
    global_state = torch.nn.modules.module
    return any(getattr(module, name, None) for name in HOOKS) or any(
        getattr(global_state, "_global" + name, None) for name in HOOKS)


def forward_function(module):
    method = module.forward
    if not isinstance(method, types.MethodType) or method.__self__ is not module:
        raise ValueError("Only ordinary bound forward methods are supported")
    return method.__func__


def _scalar(value):
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("Only scalar Python state or registered tensors may be captured")
    return (type(value).__name__, value)


def _attribute_signature(value):
    if isinstance(value,types.ModuleType): return ("module",id(value))
    if isinstance(value,torch.dtype): return ("dtype",str(value))
    if callable(value):
        defaults = tuple(_scalar(v) for v in (getattr(value,"__defaults__",None) or ()))
        keywords = tuple(sorted((k,_scalar(v)) for k,v in (getattr(value,"__kwdefaults__",None) or {}).items()))
        return ("callable",id(value),id(getattr(value,"__code__",None)),defaults,keywords)
    return _scalar(value)


class ModuleGuard:
    def __init__(self, module, inputs, *, scalar_names, callees, methods, attributes):
        self.original = module
        self.input_specs = tuple(tensor_spec(t) for t in inputs)
        self.input_keys = tuple(tensor_key(t) for t in inputs)
        self.fn = forward_function(module)
        self.code = self.fn.__code__
        if self.fn.__closure__ or self.fn.__defaults__ or self.fn.__kwdefaults__:
            raise ValueError("Forward closures and default arguments are unsupported")
        self.scalar_names = tuple(sorted(scalar_names))
        self.globals = tuple(sorted(i.argval for i in dis.get_instructions(self.fn) if i.opname == "LOAD_GLOBAL"))
        self.callees = tuple(callees)
        self.methods = tuple(methods)
        self.attributes = tuple(sorted(attributes))
        self.state = self._state()
        self.env = tuple(os.environ.get(k) for k in FLAGS)
        self.default_dtype = torch.get_default_dtype()

    def _state(self):
        m = self.original
        if any(n in vars(m) for n in ("forward","_call_impl","_slow_forward")) or getattr(m,"_compiled_call_impl",None) is not None:
            raise ValueError("Instance invocation overrides and compiled module calls are unsupported")
        if m._modules or hooks_present(m) or m.training:
            raise ValueError("Child modules, hooks, and training mode are unsupported")
        fn = forward_function(m)
        if fn is not self.fn or fn.__code__ is not self.code or fn.__defaults__ or fn.__kwdefaults__:
            raise ValueError("Forward implementation changed")
        registered = _registered_tensors(m)
        tensors = tuple((name,id(t),tensor_key(t),t.requires_grad) for name,t in registered)
        scalars = tuple((n, _scalar(getattr(m, n))) for n in self.scalar_names)
        globals_state = []
        for name in self.globals:
            value = fn.__globals__.get(name)
            if isinstance(value, types.ModuleType) and value.__name__ in ("torch", "torch.nn.functional", "math"):
                globals_state.append((name, id(value)))
            else:
                globals_state.append((name, _scalar(value)))
        calls = tuple((alias, path, _attribute_signature(self._resolve(alias, path))) for alias, path in self.callees)
        methods = tuple((name, id(getattr(torch.Tensor, name))) for name in self.methods)
        invocation = tuple(id(getattr(type(m),name)) for name in ("__call__","__getattribute__","__getattr__","_call_impl","_slow_forward","_compiled_call_impl"))
        names = set(self.scalar_names) | set(m._parameters) | set(m._buffers)
        descriptors = tuple((name,_descriptor(type(m),name)) for name in sorted(names))
        attributes = tuple((alias,path,_attribute_signature(self._resolve(alias,path))) for alias,path in self.attributes)
        return tensors, scalars, tuple(globals_state), calls, methods, invocation, descriptors, attributes

    def _resolve(self, alias, path):
        value = self.fn.__globals__[alias]
        for name in path:
            value = getattr(value, name)
        return value

    def matches(self, inputs):
        try:
            if tuple(os.environ.get(k) for k in FLAGS) != self.env or torch.get_default_dtype() != self.default_dtype:
                return False
            if torch.jit.is_tracing(): return False
            if torch.is_autocast_enabled("mps") or torch.is_autocast_enabled("cpu"):
                return False
            if tuple(tensor_key(t) for t in inputs) != self.input_keys or self._state() != self.state:
                return False
            if torch.is_grad_enabled() and any(t.requires_grad for t in (*inputs, *(t for _,t in _registered_tensors(self.original)))):
                return False
            return True
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
            return False
