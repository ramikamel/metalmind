"""Presentation text is derived from checked evidence, with explicit timing labels."""
import unicodedata


def safe_text(value):
    return "".join(c for c in str(value) if c in "\n\t" or not unicodedata.category(c).startswith("C"))


def build_chapters(data):
    pilot=data["pilot"];hardware=data["hardware"]
    chapters=[{"title":"MetalMind","label":"SAVED EVIDENCE + OPTIONAL FRESH CORRECTNESS CHECK","lines":[
        "A Meta-powered agent for verified Apple GPU kernel optimization.",
        "Model proposes Metal -> compile -> verify -> benchmark -> feedback.",
        "Host code owns fixtures, numerical decisions and promotion.",
        "No model generation is performed in this playback."]}]
    chapters.append({"title":"The original FP32 pilot","label":"RECORDED / FP32 / ASSUMED DOMAIN","lines":[
        f"{pilot['jobs']} signatures: {pilot['promoted']} promoted; {pilot['no_win']} no-win.",
        f"{pilot['generation_requests']} model requests; all outcomes stay in the denominator.",
        "Bias + tanh GELU, RMSNorm and stable row softmax.",
        f"Target: {hardware['model']}, {hardware['cores']} GPU cores.",
        "Checked mode is the public default; these gains assume the input domain."]})
    if data.get("feedback_example"):
        lines=[]
        for index,event in enumerate(data["feedback_example"],1):
            reason=event.get("reason") or f"{event.get('public_cases')} public numerical cases passed"
            lines.append(f"Candidate {index} ({event['candidate_id'][:12]}): {event['status']} — {reason}.")
        lines.extend(["These are saved evaluator results from actual Meta proposals.","Search results are provisional; private and timing gates follow."])
        chapters.append({"title":"The real feedback loop","label":"RECORDED SEARCH / NOT LIVE GENERATION","lines":lines})
    for winner in sorted(data["function_winners"],key=lambda w:w["case"]):
        compiled=[c for c in winner["comparisons"] if "inductor" in c["baseline"]]
        title=f"{winner['family']} / {winner['case']}"
        lines=[f"Shape {tuple(winner['shape'])}; {winner['private_cases']} private numerical cases passed."]
        for c in compiled: lines.append(f"Session {c['session']}: {c['speedup']:.3f}x vs Inductor fullgraph; lower 95% {c['lower']:.3f}x.")
        lines.extend(["Public-call timing includes guards, allocation, dispatch and completion.",f"Kernel source SHA-256: {winner['source_hash'][:20]}..."])
        chapters.append({"title":title,"label":"RECORDED FUNCTION CONFIRMATIONS","lines":lines})
    selected=next(w for w in data["function_winners"] if w["artifact_id"]==data["selected_artifact_id"])
    chapters.append({"title":"Meta-generated Metal","label":"SAVED SOURCE EXCERPT / VERIFIED SOURCE HASH",
        "lines":[f"Immutable source: {selected['source_hash'][:24]}...","Full source is preserved with its numerical and timing evidence."],
        "code":selected["kernel_preview"][:1000]})
    chapters.append({"title":"Whole-module gains need context","label":"SEPARATE WHOLE-MODULE EXPERIMENTS","lines":[
        "A flat residual-activation module has its own correctness/timing gate.",
        "Historical eager-only sessions: "+" / ".join(f"{c['speedup']:.3f}x" for c in data['module']['confirmations'])+".",
        "A local kernel gain is not automatically a strongest whole-model gain."]})
    comparison=data.get("compiled_module_comparison")
    if comparison:
        lines=[]
        for c in comparison["comparisons"]:
            label="compiled whole module" if "inductor" in c["baseline"] else "eager whole module"
            lines.append(f"Session {c['session']} vs {label}: "+(f"{c['speedup']:.3f}x" if c['status']=='valid' else "inconclusive"))
        lines.extend(["Below 1.0x means the regional rewrite is slower than that baseline.",
            "Module promotion now requires every verified available baseline."])
        gate=data.get("module_gate")
        if gate: lines.append(f"Current module gate: {gate['outcome']}; optimized rewrite enabled: {gate['enabled']}.")
        chapters.append({"title":"The compiler counterexample","label":"RECORDED / WHOLE-MODULE BASELINE CONTEXT","lines":lines})
    else:
        chapters[-1]["lines"].append("A compiled whole-module comparison is unavailable in this bundle.")
    live=data.get("live_check")
    if live:
        lines=[f"Outcome: {live['outcome']}"]
        if live['outcome']=='passed':
            lines.extend(["New input values matched the original PyTorch function.","Changed shapes and gradient mode used the original fallback.",f"Maximum absolute error: {live['max_absolute_error']:.3g}."])
        else: lines.append(live['reason'])
        chapters.append({"title":"Offline reuse check","label":"FRESH CORRECTNESS / NO FRESH PERFORMANCE CLAIM","lines":lines})
    chapters.append({"title":"Scope and next frontier","label":"LIMITS / ZERO MODEL REQUESTS IN THIS REPLAY","lines":[
        "Exact shapes and supported pure forward patterns on the recorded chip.",
        "No arbitrary nested-model, training or cross-chip speedup claim.",
        "This replay covers the original FP32 pilot. See the repo for later FP32 and FP16 results.",
        "This is saved evidence, not live generation or a recording of search latency.",
        "Source, raw pairs, intervals and no-win outcomes are public."]})
    return [{k:([safe_text(v) for v in value] if isinstance(value,list) else safe_text(value)) for k,value in chapter.items()} for chapter in chapters]
