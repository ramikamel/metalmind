import json
_MATH={
"bias_gelu":"z=float(x)+float(parameter[column]); y=0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715*z*z*z))). Tanh GELU, not erf.",
"rmsnorm":"y=x*rsqrt(mean_over_last_axis(x*x)+1e-5)*parameter[column]. No centering.",
"softmax":"Subtract each row maximum, exponentiate, sum, divide. Normalize final axis."
}

def build_prompt(context):
    w=context.workload;dtype="float" if w.inputs[0].dtype=="float32" else "half"
    args=[f"device const {dtype}* x"]
    if w.family!="softmax": args.append(f"device const {dtype}* parameter")
    args.extend([f"device {dtype}* out","constant uint& B","constant uint& D","constant float& epsilon"])
    return "\n".join([
        "Optimize a pure PyTorch forward function for an Apple Metal GPU.",
        'Return ONLY JSON with schema_version=1, entry_point="metalmind_kernel", source, threads (3 integers), group_size (3 integers), hypothesis (short testable explanation). No other fields.',
        "Use #include <metal_stdlib> and using namespace metal. No other includes or macros; no Python, paths, shell, or dynamic scratch.",
        f"ABI: kernel void metalmind_kernel({', '.join(args)}, ... Metal thread attributes ...).",
        "Buffers are contiguous row-major. Tensor parameters are runtime inputs, never constants. All B*D output elements must be written, inputs read-only.",
        "One entry point, one dispatch. No assumed SIMD width. Start with a simple correct implementation, conservative threadgroups, bounds checks and uniform barriers.",
        "Use FP32 intermediates and standard math; cast only final output to its specified dtype.",
        "Semantics: "+_MATH[w.family],
        "Workload: "+w.model_dump_json(),
        "Finite normal values or zero. Input bound: "+str({"bias_gelu":10,"rmsnorm":100,"softmax":80}[w.family])+". Parameter bound: 10.",
        f"Requests remaining: {context.remaining_requests}; candidates: {context.remaining_candidates}.",
        "Incumbent source: "+(context.incumbent_source or "None"),
        "Public feedback: "+json.dumps(context.feedback)])
