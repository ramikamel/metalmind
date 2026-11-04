"""Render a portable HTML replay of selected MetalMind experiments."""

import argparse
from html import escape
import json
from pathlib import Path

FP32_PILOT = "2026-11-02-fp32-pilot"
FP32_TRACE = Path(FP32_PILOT) / "jobs/bias_gelu-large"
CHECKED = "2026-11-03-checked-fp32-full"
FP16_PILOT = "2026-11-03-assumed-fp16-pilot"
RMSNORM = "2026-11-03-fp16-large-rmsnorm-resolution"
FP16_RECHECK = "2026-11-03-fp16-confirmation-recheck"
BASELINES = {
    "original_eager": "Eager PyTorch",
    "functional_rms_norm": "PyTorch functional RMSNorm",
    "torch_compile_inductor_fullgraph": "Inductor fullgraph",
}
OUTCOMES = {"promoted": "Promoted", "no_win": "No win"}

STYLES = """
:root {
  color-scheme: dark;
  font: 16px/1.55 system-ui, sans-serif;
  --ink: #edf4fb; --muted: #b2c2d5; --accent: #88ead1;
  --line: #344358; --panel: #151f30; --warning: #ffbe91;
}
* { box-sizing: border-box; }
html { scroll-padding-top: 1rem; }
body { margin: 0; color: var(--ink); background: #0b1220; }
main, footer { width: min(71rem, calc(100% - 2rem)); margin-inline: auto; }
header { padding-block: clamp(2.5rem, 8vw, 6rem) 2rem; }
h1, h2, h3, p { margin: 0; }
h1 { font-size: clamp(3rem, 9vw, 7rem); line-height: .95; letter-spacing: -.075em; }
h2 { font-size: clamp(1.55rem, 4vw, 2.35rem); line-height: 1.15; letter-spacing: -.03em; }
h3 { font-size: 1.12rem; }
a { color: var(--accent); text-underline-offset: .22em; }
:focus-visible { outline: 3px solid var(--accent); outline-offset: 4px; }
.eyebrow { color: var(--accent); font: 700 .83rem/1.4 ui-monospace, monospace;
  letter-spacing: .12em; text-transform: uppercase; }
header h1 { margin-top: .55rem; }
.lede { max-width: 52ch; font-size: clamp(1.15rem, 3vw, 1.5rem); margin-top: 1rem; }
.context, .muted, .evidence { color: var(--muted); }
.context { margin-top: .75rem; }
nav { display: flex; flex-wrap: wrap; gap: .6rem 1.3rem; margin-top: 1.5rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 17rem), 1fr));
  gap: .8rem; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: .85rem;
  padding: 1rem 1.2rem; min-width: 0; }
.kpi { font-size: clamp(1.8rem, 4vw, 2.9rem); font-weight: 750; line-height: 1.1;
  letter-spacing: -.055em; font-variant-numeric: tabular-nums; margin-block: .25rem; }
.good { color: var(--accent); }
.warn { color: var(--warning); }
section { border-top: 1px solid var(--line); padding-block: 2.1rem; }
section h2 { margin-block: .35rem .5rem; }
section > p + p, .card p + p { margin-top: .55rem; }
.trace { list-style: none; padding: 0; margin-block: 1rem; }
.trace li { margin: 0; }
.trace h3 { margin-block: .4rem .55rem; }
details { border-top: 1px solid var(--line); margin-top: .85rem; padding-top: .65rem; }
summary { cursor: pointer; font-weight: 650; color: var(--accent); }
pre { max-height: 23rem; overflow: auto; padding: .8rem; background: #0b1220;
  border: 1px solid var(--line); border-radius: .4rem; }
code { font: .9rem/1.6 ui-monospace, monospace; }
.evidence { font-size: .9rem; overflow-wrap: anywhere; margin-top: .7rem; }
.table-wrap { overflow: auto; margin-block: .8rem; }
table { border-collapse: collapse; width: 100%; text-align: left; }
caption { text-align: left; color: var(--muted); padding-bottom: .45rem; }
th, td { padding: .65rem .65rem; border-bottom: 1px solid var(--line); vertical-align: top; }
thead th { color: var(--muted); font-size: .84rem; }
td strong { font-variant-numeric: tabular-nums; }
td small { display: block; color: var(--muted); white-space: nowrap; }
.decisive { background: #192a38; }
footer { border-top: 1px solid var(--line); color: var(--muted); padding-block: 1rem 2rem; }
@media (max-width: 30rem) {
  main, footer { width: calc(100% - 1.35rem); }
  th, td { padding-inline: .35rem; }
}
"""


def read_text(root, path):
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Evidence path escapes its archive: {path}")
    return resolved.read_text(encoding="utf-8")


def read_json(root, path):
    return json.loads(read_text(root, path))


def shape_of(manifest):
    return " × ".join(str(int(size)) for size in manifest["workload"]["inputs"][0]["shape"])


def evidence_path(path):
    return f'<p class="evidence">Archive <code>results/{escape(str(path))}</code></p>'


def render_candidates(root, manifest):
    events = (
        json.loads(line)
        for line in read_text(root, FP32_TRACE / "events.jsonl").splitlines()
    )
    proposals = {
        item["candidate"]["candidate_id"]: item["candidate"]["proposal"]
        for item in manifest["archive"]
    }
    model = ""
    cards = []
    for event in events:
        if event["event"] == "model_response":
            model = event["model"]
        if event["event"] != "evaluation":
            continue
        candidate_id = event["candidate_id"]
        source = read_text(root, FP32_TRACE / "candidates" / f"{candidate_id}.metal")
        result = event["result"]
        verification = result["verification"]
        if verification["passed"]:
            verdict = "Passed public checks"
            badge = "good"
            description = f'{int(verification["cases"])} public correctness cases passed.'
        else:
            verdict = "Rejected"
            badge = "warn"
            description = escape(verification["reason"] or result["reason"])
        proposal = proposals.get(candidate_id)
        hypothesis = (
            f'<p>{escape(proposal["hypothesis"])}</p>' if proposal else ""
        )
        cards.append(f"""
          <li class="card">
            <p class="eyebrow">{len(cards) + 1:02} · {escape(model)}</p>
            <h3 class="{badge}">{verdict}</h3>
            <p class="muted">{description}</p>
            {hypothesis}
            <details>
              <summary>Inspect archived Metal source</summary>
              <pre tabindex="0" aria-label="Metal source"><code>{escape(source)}</code></pre>
              <p class="evidence">Candidate <code>{escape(candidate_id)}</code></p>
            </details>
          </li>
        """)
    return "".join(cards)


def confirmation_sessions(root, run):
    return [
        read_json(root, run / f"confirmation-{index}.json")
        for index in (1, 2)
    ]


def comparisons(session):
    return {item["name"]: item["summary"] for item in session["timing"]["comparisons"]}


def render_confirmations(sessions):
    first, second = map(comparisons, sessions)
    if first.keys() != second.keys():
        raise ValueError("Evidence sessions compare different baselines")
    # The smallest speed ratio in a session is the hardest baseline to beat.
    decisive = {min(items, key=lambda name: items[name]["speedup"]) for items in (first, second)}
    rows = []
    for name in first:
        cls = ' class="decisive"' if name in decisive else ""
        label = escape(BASELINES.get(name, name))
        cells = "".join(
            f'<td><strong>{item["speedup"]:.3f}×</strong>'
            f'<small>{item["lower"]:.3f}–{item["upper"]:.3f}×</small>'
            f'<small>{int(item["pairs"])} pairs</small></td>'
            for item in (first[name], second[name])
        )
        rows.append(f'<tr{cls}><th scope="row">{label}</th>{cells}</tr>')
    return f"""
      <div class="table-wrap" tabindex="0" role="region" aria-label="Timing confirmations">
        <table>
          <caption>Median paired speed ratio and 95% interval. Above 1.00× favors the candidate.</caption>
          <thead><tr><th scope="col">Verified baseline</th><th scope="col">Session 1</th>
            <th scope="col">Session 2</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
      </div>
    """


def build_showcase(root):
    root = root.resolve()
    trace = read_json(root, FP32_TRACE / "manifest.json")
    trace_outcome = read_json(root, FP32_TRACE / "result.json")["outcome"]
    trace_cases = read_json(root, FP32_TRACE / "private-verification.json")["verification"]["cases"]
    hardware = read_json(root, Path("2026-11-02-module-optimizer/release.json"))["hardware"]
    checked = read_json(root, Path(CHECKED) / "summary.json")
    fp16 = read_json(root, Path(FP16_PILOT) / "summary.json")
    softmax = next(row for row in fp16["rows"] if row["family"] == "softmax" and row["case"] == "medium")
    softmax_run = Path(FP16_PILOT) / "raw" / softmax["run_path"]
    softmax_shape = shape_of(read_json(root, softmax_run / "manifest.json"))
    softmax_sessions = confirmation_sessions(root, softmax_run)
    strongest_softmax = min(item["speedup"] for session in softmax_sessions
                            for item in comparisons(session).values())
    followed_up = read_json(root, Path(FP16_RECHECK) / "summary.json")
    wins = {(row["family"], row["case"]) for report in (fp16, followed_up)
            for row in report["rows"] if row["outcome"] == "promoted"}


    rms_path = Path(RMSNORM)
    rms = read_json(root, rms_path / "summary.json")
    rms_sessions = [
        read_json(root, rms_path / attempt["record"])["result"]
        for attempt in rms["attempts"] if attempt["status"] == "valid"
    ]
    rms_cases = read_json(root, rms_path / "private-verification.json")["verification"]["cases"]
    rms_protocol = read_json(root, rms_path / "protocol.json")
    rms_run = Path(FP16_PILOT) / "raw" / rms_protocol["source_run"]
    rms_shape = shape_of(read_json(root, rms_run / "manifest.json"))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
  <title>MetalMind · Evidence replay</title>
  <style>{STYLES}</style>
</head>
<body>
<main>
  <header>
    <p class="eyebrow">Saved evidence replay · zero model requests to view</p>
    <h1>MetalMind</h1>
    <p class="lede">Muse Wonder writes Metal kernels for PyTorch. MetalMind checks their output and races them against verified PyTorch baselines.</p>
    <p class="context">{escape(hardware["model"])} · {int(hardware["cores"])}-core GPU · recorded runs</p>
    <nav aria-label="Replay chapters">
      <a href="#search">01 · Search trace</a>
      <a href="#modes">02 · Measurement modes</a>
      <a href="#rmsnorm">03 · No-win decision</a>
    </nav>
  </header>
  <section class="grid" aria-label="Results at a glance">
    <div class="card">
      <p class="eyebrow">Checked FP32</p>
      <p class="kpi">{int(checked["promoted"])} / {int(checked["attempted"])}</p>
      <p class="muted">Campaign jobs promoted · default input checks included</p>
    </div>
    <div class="card">
      <p class="eyebrow">Assumed-domain FP16</p>
      <p class="kpi good">{strongest_softmax:.2f}×</p>
      <p class="muted">Medium softmax vs fastest verified baseline across both sessions</p>
      <p class="muted">{len(wins)} / {int(fp16["attempted"])} promoted across the pilot and follow-up · {int(fp16["promoted"])} / {int(fp16["attempted"])} in the original pilot</p>
    </div>
    <div class="card">
      <p class="eyebrow">Large FP16 RMSNorm</p>
      <p class="kpi warn">{OUTCOMES[rms["outcome"]]}</p>
      <p class="muted">Final confirmation kept the PyTorch path</p>
    </div>
  </section>
  <section id="search">
    <p class="eyebrow">01 · Search trace</p>
    <h2>Two generated kernels, one promotion</h2>
    <p class="muted">Large FP32 bias/GELU · {shape_of(trace)} · assumed domain</p>
    <ol class="grid trace">{render_candidates(root, trace)}</ol>
    <p><strong class="good">{OUTCOMES[trace_outcome]}</strong> after {int(trace_cases)} private correctness cases and two independent timing sessions.</p>
    {render_confirmations(confirmation_sessions(root, FP32_TRACE))}
    {evidence_path(FP32_TRACE)}
  </section>
  <section id="modes">
    <p class="eyebrow">02 · Measurement modes</p>
    <h2>The caller decides which checks run</h2>
    <p><strong>Checked mode</strong> includes runtime checks for finite, bounded, nonsubnormal inputs. <strong>Assumed domain</strong> is an opt-in for callers who guarantee those conditions.</p>
    <p class="muted">The campaigns use different dtypes, input-check policies and trial designs. Their promotion counts describe separate experiments.</p>
    {evidence_path(Path(CHECKED) / "summary.json")}
    <details>
      <summary>Inspect the FP16 medium softmax win · {softmax_shape}</summary>
      {render_confirmations(softmax_sessions)}
      {evidence_path(softmax_run)}
    </details>
  </section>
  <section id="rmsnorm">
    <p class="eyebrow">03 · No-win decision</p>
    <h2>Large FP16 RMSNorm stays on PyTorch</h2>
    <p class="muted">{rms_shape} · assumed domain · {int(rms_cases)} private correctness cases passed</p>
    {render_confirmations(rms_sessions)}
    <p><strong class="warn">{OUTCOMES[rms["outcome"]]}.</strong> Promotion requires at least 1.05× and a lower interval bound above 1.00× against every verified baseline in both sessions. Inductor was the decisive baseline.</p>
    {evidence_path(rms_path)}
  </section>
</main>
<footer>Saved MetalMind experiments · Source and timing records are archived in the repository.</footer>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output", type=Path, required=True, help="Path to the standalone HTML file")
    args = parser.parse_args()
    try:
        page = build_showcase(args.evidence_root)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.exit(1, f"Could not read showcase evidence: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [line if line.strip() else "" for line in page.splitlines()]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
