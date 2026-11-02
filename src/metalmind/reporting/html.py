"""Standalone escaped evidence report; generated text never becomes markup."""
from html import escape
import json
from pathlib import Path


def _text(value):
    return escape(str(value))


def _json(value):
    return "<pre>" + escape(json.dumps(value, indent=2, allow_nan=False)) + "</pre>"


def _comparison_rows(reports):
    rows = []
    for label, report in reports:
        timing = report.get("timing") or {}
        for comparison in timing.get("comparisons", []):
            summary = comparison["summary"]
            values = (label, comparison["name"], f"{summary['speedup']:.3f}×",
                f"{summary['lower']:.3f}–{summary['upper']:.3f}×", summary["pairs"])
            rows.append("<tr>" + "".join("<td>" + _text(v) + "</td>" for v in values) + "</tr>")
    if not rows:
        return "<p>No valid timing comparisons were saved for this stage.</p>"
    return ('<div class="table"><table><thead><tr><th>Session / candidate</th><th>Baseline</th>'
        '<th>Median speedup</th><th>95% interval</th><th>Pairs</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>")


def render_report(summary, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = summary.get("result") or {}
    manifest = summary.get("manifest") or {}
    workload = manifest.get("workload") or {}
    config = manifest.get("config") or {}
    stages = summary.get("stages") or {}
    final = summary.get("final_evidence") or {}
    outcome = str(result.get("outcome", manifest.get("phase", "Run report")))
    title = "MetalMind — " + outcome
    search = [(e.get("candidate_id", "unavailable")[:12], e["result"])
        for e in summary.get("events", []) if e["event"] == "evaluation"]
    confirmations = [(f"Session {i}", report)
        for i, report in enumerate(final.get("confirmations", []), 1)]
    cards = [("Outcome", outcome), ("Model requests", manifest.get("requests", "unavailable")),
        ("Output tokens", manifest.get("output_tokens", "unavailable")),
        ("Domain mode", config.get("domain_mode", "unavailable"))]
    content = ['<header><p class="eyebrow">METALMIND / RUN EVIDENCE</p><h1>' + _text(outcome)
        + '</h1><p>' + _text(result.get("reason", "Run has not finalized")) + '</p></header>']
    content.append('<div class="cards">' + "".join('<div class="card"><span>' + _text(k)
        + '</span><strong>' + _text(v) + '</strong></div>' for k, v in cards) + '</div>')
    content.append('<p class="notice">Speedups describe the recorded public-call boundary and domain mode. '
        'Search measurements are not confirmed promotion. Compilation and model time are outside warm timing.</p>')
    content.append('<section><h2>Workload and stages</h2>' + _json({"workload": workload or "unavailable",
        "stages": stages or "unavailable", "candidate_counts": summary.get("candidate_counts", {})}) + '</section>')
    content.append('<section><h2>Confirmation sessions</h2>' + _comparison_rows(confirmations))
    for label, report in confirmations:
        if report.get("status") != "valid":
            content.append('<p>' + _text(label + ": " + str(report.get("reason") or report.get("status"))) + '</p>')
    content.append('<details><summary>Private numerical verification</summary>'
        + _json(final.get("private_verification")) + '</details></section>')
    content.append('<section><h2>Search evaluations</h2>' + _comparison_rows(search))
    for label, report in search:
        content.append('<p><code>' + _text(label) + '</code> · ' + _text(report.get("status"))
            + ' · ' + _text(report.get("reason") or "See raw evidence for numerical and timing details") + '</p>')
    content.append('</section><section><h2>Candidate source</h2>')
    for candidate in summary.get("candidates", []):
        content.append('<details><summary>' + _text(candidate["candidate_id"]) + '</summary><pre>'
            + _text(candidate["source"]) + '</pre></details>')
    content.append('</section><section><details><summary>Complete raw evidence and configuration</summary>'
        + _json(summary) + '</details></section>')
    css = """
    :root{color-scheme:light;font-family:system-ui,-apple-system,sans-serif;color:#172b31;background:#f3f6f4}
    body{max-width:1160px;margin:3rem auto;padding:0 1.5rem;line-height:1.55}
    h1{font-size:3.5rem;line-height:1.1;letter-spacing:-.06em;margin:.5rem 0 1rem}
    h2{font-size:1.25rem;letter-spacing:-.02em}.eyebrow{font-size:.75rem;letter-spacing:.15em;color:#356654}
    header{max-width:850px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1rem;margin:2rem 0}
    .card,section{background:#fff;border:1px solid #dbe4de;border-radius:12px;padding:1.25rem}
    .card span{display:block;color:#5c6f68;font-size:.85rem}.card strong{display:block;font-size:1.5rem;margin-top:.35rem}
    section{margin:1rem 0}.notice{border-left:3px solid #448a6f;padding:.5rem 1rem;color:#3a574c}
    .table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{text-align:left;padding:.75rem;border-bottom:1px solid #e4eae6;white-space:nowrap}
    th{color:#52675d;font-weight:600}pre{font-size:.8rem;background:#f6f8f7;border:1px solid #e5ebe7;padding:1rem;border-radius:7px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}
    details{margin:1rem 0}summary{cursor:pointer;color:#275e49}code{font-size:.85em}
    @media(max-width:600px){body{margin:1.5rem auto;padding:0 1rem}h1{font-size:2.5rem}}
    """
    path.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>' + _text(title)
        + '</title><style>' + css + '</style></head><body><main>' + "".join(content) + '</main></body></html>')
    return path
