"""Read-only checks of curated historical evidence; never execute saved host code."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from metalmind.artifacts.export import verify_artifact_evidence,digest
from metalmind.benchmarking.statistics import summarize_pairs
from metalmind.contracts import TimingSeries

PILOT="2026-11-02-fp32-pilot"
MODULE="2026-11-02-module-optimizer"
COMPILED="2026-11-02-module-baselines"


def child(root, *parts):
    root=Path(root).resolve();path=root.joinpath(*parts).resolve()
    if not path.is_relative_to(root): raise ValueError("Evidence path points outside its directory")
    return path


def read_json(root,*parts):
    path=child(root,*parts)
    if path.stat().st_size>5_000_000: raise ValueError("Evidence file exceeds the supported size")
    return json.loads(path.read_text())


def identifier(value,length):
    if not isinstance(value,str) or re.fullmatch("[a-f0-9]{"+str(length)+"}",value) is None:
        raise ValueError("Invalid evidence identifier")
    return value


def validate_pairs(report):
    series=TimingSeries.model_validate(report["series"])
    if len(series.baseline_ms)<30 or len(series.order)!=len(series.baseline_ms) or series.repetitions<1:
        raise ValueError("Incomplete raw confirmation pairs")
    counts=Counter(series.order)
    if set(counts)-{"baseline_first","candidate_first"} or abs(counts["baseline_first"]-counts["candidate_first"])>1:
        raise ValueError("Raw confirmation ordering is not balanced")
    resamples=report["bootstrap_resamples"]
    if type(resamples) is not int or not 1000<=resamples<=100000: raise ValueError("Invalid bootstrap size")
    measured=summarize_pairs(series.baseline_ms,series.candidate_ms,seed=report["seed"],resamples=resamples)
    if report["summary"]["pairs"]!=measured.pairs or any(not math.isclose(report["summary"][k],getattr(measured,k),rel_tol=1e-10,abs_tol=1e-12) for k in ("speedup","lower","upper")):
        raise ValueError("Saved summary does not match raw timing pairs")
    return measured.model_dump(mode="json")


def validate_module_candidate(folder,result):
    folder=Path(folder)
    identity=read_json(folder,"candidate-identity.json")
    implementation=child(folder,"implementation")
    sources={p.name:child(implementation,p.name).read_text() for p in sorted(implementation.glob("*.py"))}
    if not sources: raise ValueError("Missing module implementation snapshots")
    actual=hashlib.sha256(json.dumps({**identity,"implementation":sources},sort_keys=True).encode()).hexdigest()
    if actual!=result["candidate_id"]: raise ValueError("Module candidate identity mismatch")
    if identity["rewritten_graph"]!=read_json(folder,"candidate-graph.json")["source"]:
        raise ValueError("Exported module graph differs from its identity")
    if identity["original_graph"]!=result["analysis"]["graph_hash"]:
        raise ValueError("Original module graph identity mismatch")
    if list(identity["artifact_ids"])!=result["artifact_ids"]: raise ValueError("Module artifact identities differ")
    private=read_json(folder,"private-verification.json")
    if private!=result["private_verification"] or private["passed"] is not True or type(private["cases"]) is not int or private["cases"]<8:
        raise ValueError("Private whole-module verification is missing or failed")
    return identity


def _load_demo_evidence(evidence_root):
    root=Path(evidence_root).resolve();pilot=child(root,PILOT);module=child(root,MODULE)
    campaign=read_json(pilot,"campaign.json");analysis=read_json(pilot,"analysis.json");summary=read_json(pilot,"summary.json")
    if campaign.get("manifest_version",1)==1:
        expected=digest(campaign["config"])
    else:
        expected=digest({k:campaign[k] for k in ("manifest_version","config","workloads")})
    if expected!=campaign["campaign_id"]: raise ValueError("Campaign identity mismatch")
    config=campaign["config"]
    expected_jobs=len(config["families"])*len(config["cases"])*len(config["strategies"])*config["trials"]
    rows=analysis["rows"]
    if len(rows)!=expected_jobs or campaign["jobs"]!=expected_jobs or summary["attempted"]!=expected_jobs:
        raise ValueError("Incomplete pilot denominator")
    seen=set();outcomes=Counter();statuses=Counter();requests=0;winners=[];artifacts={};job_rows=[];model_names=set();feedback=[]
    summary_rows={(r["family"],r["case"],r["strategy"],r["trial"]):r for r in summary["rows"]}
    for row in rows:
        key=(row["family"],row["case"],row["strategy"],row["trial"])
        if key in seen or row["family"] not in config["families"] or row["case"] not in config["cases"] or row["strategy"] not in config["strategies"] or not 0<=row["trial"]<config["trials"]:
            raise ValueError("Duplicate or unsupported pilot job")
        seen.add(key)
        if any(summary_rows[key][k]!=row[k] for k in ("outcome","requests","speedup")):
            raise ValueError("Pilot summary rows disagree")
        folder=child(pilot,"jobs",row["family"]+"-"+row["case"])
        manifest=read_json(folder,"manifest.json");result=read_json(folder,"result.json")
        if manifest["phase"]!="terminal" or result["outcome"]!=row["outcome"] or manifest["requests"]!=row["requests"]:
            raise ValueError("Pilot job is incomplete or inconsistent")
        events=[json.loads(line) for line in child(folder,"events.jsonl").read_text().splitlines()]
        if sum(e["requests"] for e in events if e["event"]=="model_response")!=row["requests"]:
            raise ValueError("Saved request accounting differs")
        for event in events:
            if event["event"]=="evaluation":
                statuses[event["result"]["status"]]+=1
                if row["family"]=="bias_gelu" and row["case"]=="large":
                    r=event["result"]
                    feedback.append({"candidate_id":event["candidate_id"],"status":r["status"],"reason":r.get("reason") or (r.get("verification") or {}).get("reason"),
                        "public_cases":(r.get("verification") or {}).get("cases")})
            if event["event"]=="model_response": model_names.add(event["model"])
        requests+=row["requests"];outcomes[row["outcome"]]+=1
        job_rows.append({"family":row["family"],"case":row["case"],"shape":manifest["workload"]["inputs"][0]["shape"],"outcome":row["outcome"]})
        if row["outcome"]=="promoted":
            artifact_id=identifier(row["artifact_id"],64);artifact=child(pilot,"artifacts",artifact_id)
            for name in ("manifest.json","evidence.json","kernel.metal"): child(artifact,name)
            verified=verify_artifact_evidence(artifact)
            if verified.artifact_id!=artifact_id: raise ValueError("Artifact directory identity differs")
            if verified.workload.reference_id!=verified.workload.family: raise ValueError("Demo requires registered semantic references")
            if verified.workload.model_dump(mode="json")!=manifest["workload"]: raise ValueError("Winner belongs to a different workload")
            if verified.domain_mode!=config["domain_mode"]: raise ValueError("Winner domain mode differs from campaign")
            evidence=read_json(artifact,"evidence.json")
            comparisons=[{"baseline":c["name"],"session":i+1,**c["summary"]} for i,s in enumerate(evidence["confirmations"]) for c in s["comparisons"]]
            if not math.isclose(min(c["speedup"] for c in comparisons),row["speedup"],rel_tol=1e-12): raise ValueError("Pilot winner score differs")
            item={"family":row["family"],"case":row["case"],"shape":list(verified.workload.inputs[0].shape),"domain_mode":verified.domain_mode,
                "artifact_id":artifact_id,"source_hash":verified.source_hash,"private_cases":evidence["verification"]["cases"],"comparisons":comparisons,
                "kernel_preview":"\n".join(verified.candidate.proposal.source.splitlines()[:16])}
            winners.append(item);artifacts[artifact_id]=verified
    if outcomes["promoted"]!=analysis["promoted"] or requests!=analysis["generation_requests"] or dict(statuses)!=analysis["candidate_status_counts"]:
        raise ValueError("Pilot aggregate differs from job evidence")
    release=read_json(module,"release.json");run_id=identifier(release["primary_run"],32)
    folder=child(module,run_id);result=read_json(folder,"result.json")
    validate_module_candidate(folder,result)
    if result["candidate_id"]!=release["candidate_id"] or result["outcome"]!="promoted" or result["enabled"] is not True: raise ValueError("Historical module release differs")
    if len(result["confirmations"])!=2: raise ValueError("Two historical module sessions required")
    module_sessions=[]
    for index,report in enumerate(result["confirmations"],1):
        if report!=read_json(folder,f"confirmation-{index}.json") or report["baseline"]!="original_eager_whole_module":
            raise ValueError("Module confirmation records differ")
        values=validate_pairs(report)
        if values["speedup"]<1.05 or values["lower"]<=1: raise ValueError("Historical module promotion does not pass")
        module_sessions.append(values)
    if any(i not in artifacts for i in result["artifact_ids"]): raise ValueError("Unknown selected function artifact")
    best=max(winners,key=lambda w:min(c["speedup"] for c in w["comparisons"]))
    data={"schema_version":1,"mode":"saved_evidence_replay","model_requests":0,
        "recorded_session_generation_requests":release["session_generation_requests"],"hardware":release["hardware"],"runtime":release["runtime"],
        "models":sorted(model_names),"feedback_example":feedback,"pilot":{"campaign_id":campaign["campaign_id"],"jobs":expected_jobs,"promoted":outcomes["promoted"],"no_win":outcomes["no_win"],
            "generation_requests":requests,"candidate_status_counts":dict(statuses),"rows":job_rows},
        "function_winners":winners,"selected_artifact_id":best["artifact_id"],
        "module":{"candidate_id":result["candidate_id"],"graph_hash":result["analysis"]["graph_hash"],"domain_mode":release["domain_mode"],
            "baseline":"original eager whole module (historical)","private_cases":result["private_verification"]["cases"],"confirmations":module_sessions},
        "compiled_module_comparison":None,"module_gate":None,"live_check":None}
    compiled=child(root,COMPILED)
    if (compiled/"report.json").exists():
        report=read_json(compiled,"report.json")
        if report["phase"]!="terminal" or report["graph_hash"]!=data["module"]["graph_hash"] or len(report["sessions"])!=2:
            raise ValueError("Compiled module experiment is incomplete or incompatible")
        if report["candidate_verification"]["passed"] is not True: raise ValueError("Candidate verification failed")
        verified=report.get("compiled_verification")
        if verified is not None and (verified.get("passed") is not True or verified.get("cases",0)<8):
            raise ValueError("Compiled module correctness verification failed")
        if not report["artifact_ids"] or any(i not in artifacts or artifacts[i].environment_fingerprint!=report["environment_fingerprint"] for i in report["artifact_ids"]):
            raise ValueError("Compiled comparison environment differs")
        comparisons=[];sets=[]
        for index,session in enumerate(report["sessions"],1):
            if session["session"]!=index: raise ValueError("Compiled session identifiers differ")
            names=[c["baseline"] for c in session["comparisons"]]
            if len(names)!=len(set(names)): raise ValueError("Duplicate module baseline")
            sets.append(set(names))
            for c in session["comparisons"]:
                if c["status"] not in ("valid","inconclusive"): raise ValueError("Unknown measurement status")
                comparisons.append({"session":index,"baseline":c["baseline"],"status":c["status"],
                    **(validate_pairs(c) if c["status"]=="valid" else {"reason":c["reason"]})})
        if sets[0]!=sets[1] or "original_eager_whole_module" not in sets[0]: raise ValueError("Whole-module baseline sets differ")
        selected=[c for c in comparisons if "inductor" in c["baseline"]]
        outcome=("unavailable" if not selected else "inconclusive" if any(c["status"]!="valid" for c in selected)
            else "confirmed_gain" if all(c["speedup"]>=1.05 and c["lower"]>1 for c in selected) else "no_confirmed_gain")
        if selected and (len(selected)!=2 or verified is None): raise ValueError("Compiled comparison evidence is incomplete")
        if outcome!=report["compiled_comparison_outcome"]: raise ValueError("Compiled comparison outcome contradicts raw evidence")
        if report["model_requests"]!=0: raise ValueError("Offline comparison unexpectedly reports model requests")
        data["compiled_module_comparison"]={"outcome":outcome,"comparisons":comparisons,"model_requests":0}
        gate_folder=child(compiled,"strongest-gate")
        if (gate_folder/"result.json").exists():
            gate=read_json(gate_folder,"result.json");identity=validate_module_candidate(gate_folder,gate)
            if identity.get("baseline_policy")!="all_verified_available" or gate.get("confirmation_format_version")!=2:
                raise ValueError("Strongest-baseline gate policy differs")
            required=gate["required_baselines"]
            if required!=identity["required_baselines"]: raise ValueError("Gate baseline identities differ")
            gate_comparisons=[]
            for index,session in enumerate(gate["confirmations"],1):
                if session!=read_json(gate_folder,f"confirmation-{index}.json"): raise ValueError("Gate confirmation records differ")
                for c in session["comparisons"]:
                    if c["baseline"] not in required: raise ValueError("Gate used an undeclared baseline")
                    if c["status"]=="valid": gate_comparisons.append({"session":index,"baseline":c["baseline"],**validate_pairs(c)})
            if gate["outcome"]=="no_win" and (gate["enabled"] is not False or not any(c["speedup"]<1.05 or c["lower"]<=1 for c in gate_comparisons)):
                raise ValueError("Disabled gate outcome contradicts its comparisons")
            data["module_gate"]={"outcome":gate["outcome"],"enabled":gate["enabled"],"required_baselines":required,"comparisons":gate_comparisons}
    return data


def load_demo_evidence(evidence_root):
    try: return _load_demo_evidence(evidence_root)
    except (KeyError,TypeError,IndexError,OverflowError) as exc:
        raise ValueError("Malformed demo evidence: " + str(exc)[:200]) from None
