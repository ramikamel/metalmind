"""Archive a completed campaign and reproduce its analysis without GPU access."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from metalmind.artifacts.export import verify_artifact_evidence
from metalmind.artifacts.store import atomic_json
from metalmind.demonstration.provenance import git_package_fingerprint
from scripts.report_campaign import build_campaign_report
from scripts.report_campaign_comparison import build_campaign_comparison


def _read(path):
    return json.loads(path.read_text())


def _verified_source_commit(root, expected):
    command = ["git", "rev-parse", "HEAD"]
    before = subprocess.check_output(command, cwd=root, text=True).strip()
    actual = git_package_fingerprint(root)["sha256"]
    after = subprocess.check_output(command, cwd=root, text=True).strip()
    if actual != expected or before != after:
        raise ValueError("Checkout commit does not match the frozen package source")
    return before


def _evidence_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise ValueError("Evidence paths must be relative to the execution checkout")
    source = root / relative
    if source.resolve() != source or not source.is_dir():
        raise ValueError(f"Evidence directory is missing or uses a symbolic link: {relative}")
    if any(item.is_symlink() for item in source.rglob("*")):
        raise ValueError(f"Evidence directory contains a symbolic link: {relative}")
    return relative, source


def _verify_archived_row(raw, row, frozen):
    settings = frozen["config"]
    strategy = row["strategy"]
    cap = 1 if strategy == "one_shot" else settings["run_config"]["max_requests"]
    slots = row["proposal_slots"]
    accounting = row["request_accounting"]
    if (not 0 <= slots <= cap or row["requests"] != slots
            or row["model_requests"] != (0 if strategy == "template" else slots)):
        raise ValueError("Job request accounting disagrees with the frozen per-job budget")

    run = raw / row["run_path"] if row["run_path"] else None
    manifest = _read(run / "manifest.json") if run else {}
    if slots != manifest.get("requests", 0):
        raise ValueError("Job proposal slots disagree with the copied request ledger")
    if accounting == "search_ledger":
        if run is None:
            raise ValueError("A finalized search must have a run directory")
        result = _read(run / "result.json")
        if any(row.get(key) != result.get(key) for key in ("outcome", "artifact_path", "reason")):
            raise ValueError("Job outcome disagrees with the copied final result")
    elif accounting not in ("interrupted_upper_bound", "exception_upper_bound") or row["outcome"] != "failed":
        raise ValueError("Unknown or inconsistent job request accounting")

    if accounting == "search_ledger" or "workload" in manifest:
        expected = next(item["workload"] for item in frozen["workloads"]
                        if (item["family"], item["case"]) == (row["family"], row["case"]))
        if (manifest.get("workload") != expected
                or manifest["config"]["domain_mode"] != settings["domain_mode"]
                or manifest["config"]["max_requests"] != cap):
            raise ValueError("Copied run does not match the frozen workload or configuration")

    if row["outcome"] != "promoted":
        if row["speedup"] != 1.0:
            raise ValueError("Fallback jobs must have a neutral final speedup")
        return
    artifact = raw / row["artifact_path"]
    verified = verify_artifact_evidence(artifact)
    if (verified.artifact_id != artifact.name
            or verified.workload.model_dump(mode="json") != manifest["workload"]
            or verified.domain_mode != settings["domain_mode"]):
        raise ValueError("Promoted artifact does not match the copied run or domain")
    evidence = _read(artifact / "evidence.json")
    score = min(c["summary"]["speedup"] for session in evidence["confirmations"]
                for c in session["comparisons"])
    if not math.isclose(row["speedup"], score, rel_tol=1e-12, abs_tol=0):
        raise ValueError("Promoted speedup disagrees with confirmed artifact evidence")


def archive_completed_campaign(campaign, run_root, destination):
    campaign, root, destination = Path(campaign), Path(run_root).resolve(), Path(destination)
    frozen = _read(campaign / "campaign.json")
    summary = _read(campaign / "summary.json")
    execution = frozen["execution"]
    rows = summary["rows"]
    if not summary["complete"] or summary["attempted"] != len(rows) or len(rows) != frozen["jobs"]:
        raise ValueError("Only a complete campaign can be archived")
    requests = sum(row["model_requests"] for row in rows)
    if (requests != summary["model_requests"] or not 0 <= requests <= execution["max_model_requests"]
            or any(row["model_requests"] < 0 for row in rows)):
        raise ValueError("The campaign request total disagrees with its rows or ceiling")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    commit = _verified_source_commit(root, execution["source_sha256"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".campaign-archive-", dir=destination.parent) as temporary:
        archive = Path(temporary) / "archive"
        archive.mkdir()
        shutil.copyfile(campaign / "campaign.json", archive / "campaign.json")
        shutil.copyfile(campaign / "summary.json", archive / "summary.json")
        copied = set()
        for row in rows:
            artifact = row.get("artifact_path")
            if bool(artifact) != (row["outcome"] == "promoted"):
                raise ValueError("A promoted row must identify its exported artifact")
            for name in (row["run_path"], artifact):
                if not name or name in copied:
                    continue
                relative, source = _evidence_path(root, name)
                shutil.copytree(source, archive / "raw" / relative)
                copied.add(name)
            _verify_archived_row(archive / "raw", row, frozen)
        analysis = build_campaign_report(archive, run_root=archive / "raw")
        atomic_json(archive / "analysis.json", analysis)
        comparison = build_campaign_comparison(archive, run_root=archive / "raw")
        atomic_json(archive / "comparison.json", comparison)
        atomic_json(archive / "archive.json", {
            "campaign_id": frozen["campaign_id"], "source_commit": commit,
            "source_sha256": execution["source_sha256"], "jobs": len(rows),
            "model_requests": requests, "max_model_requests": execution["max_model_requests"],
        })
        checksums = {
            p.relative_to(archive).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(archive.rglob("*")) if p.is_file()
        }
        atomic_json(archive / "sha256.json", checksums)
        archive.rename(destination)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    report = archive_completed_campaign(args.campaign, args.run_root, args.destination)
    print(json.dumps({key: report[key] for key in ("campaign_id", "attempted", "complete")}, indent=2))


if __name__ == "__main__":
    main()
