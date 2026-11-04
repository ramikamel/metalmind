from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_demo_showcase.py"


def run_showcase(output, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), *args],
        capture_output=True,
        text=True,
    )


def test_showcase_replays_the_archived_search_and_baseline_decisions(tmp_path):
    output = tmp_path / "replay.html"
    completed = run_showcase(output)
    assert completed.returncode == 0, completed.stderr
    html = output.read_text(encoding="utf-8")
    assert "Saved evidence replay" in html
    assert "muse-wonder-1.0" in html
    assert "A 2D float4-vectorized dispatch maps columns directly to the bias vector" in html
    assert "nonfinite output" in html
    assert "Checked FP32" in html and "0 / 108" in html
    assert "Assumed-domain FP16" in html and "6 / 9" in html and "8 / 9" in html and "9 / 9" in html
    assert "2.54×" in html
    assert "Inductor fullgraph" in html
    assert "1.093×" in html and "1.097×" in html
    assert "2.641×" in html and "2.544×" in html
    assert "0.997×" in html and "0.990×" in html
    assert "1.252×" in html and "1.268×" in html
    assert "Complete attention block" in html and "PyTorch scaled dot-product attention" in html
    assert "1.058×" in html and "1.067×" in html
    assert "No win" in html


def test_showcase_embeds_metal_source_as_text_and_replays_offline(tmp_path):
    output = tmp_path / "replay.html"
    completed = run_showcase(output)
    assert completed.returncode == 0, completed.stderr
    html = output.read_text(encoding="utf-8")
    assert html.count("#include &lt;metal_stdlib&gt;") == 2
    assert "<metal_stdlib>" not in html
    assert "default-src 'none'" in html
    assert "<script" not in html
    assert "https://" not in html


def test_missing_archive_leaves_existing_output_untouched(tmp_path):
    output = tmp_path / "replay.html"
    output.write_text("existing replay", encoding="utf-8")
    completed = run_showcase(output, "--evidence-root", str(tmp_path / "missing"))
    assert completed.returncode != 0
    assert "evidence" in completed.stderr.lower()
    assert output.read_text(encoding="utf-8") == "existing replay"
