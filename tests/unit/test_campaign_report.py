from scripts.report_campaign import build_campaign_report
from metalmind.artifacts.store import atomic_json


def test_campaign_report_distinguishes_search_gain_from_promotion(tmp_path):
    path = tmp_path / "campaign"
    job = {"family": "bias_gelu", "case": "small", "trial": 0, "strategy": "feedback"}
    next_job = {**job, "strategy": "independent"}
    atomic_json(path / "campaign.json", {
        "campaign_id": "example", "jobs": 2, "config": {"domain_mode": "checked"},
        "execution": {"job_order": [job, next_job]},
    })
    run = tmp_path / "job"
    atomic_json(run / "manifest.json", {"evaluated": 1, "elapsed_seconds": 18.0})
    (run / "events.jsonl").write_text(
        '{"event":"model_response","usage":{"input_tokens":20,"output_tokens":40},"error":null}\n'
        '{"event":"evaluation","result":{"status":"valid","verification":{"passed":true},'
        '"timing":{"domain_mode":"checked","comparisons":[{"name":"original_eager","summary":{"speedup":1.5}},'
        '{"name":"torch_compile_inductor_fullgraph","summary":{"speedup":1.1}}]}}}\n'
    )
    atomic_json(path / "summary.json", {
        "campaign_id": "example", "attempted": 1, "complete": False,
        "rows": [{**job, "outcome": "no_win", "speedup": 1.0, "model_requests": 1, "run_path": "job"}],
    })

    report = build_campaign_report(path, run_root=tmp_path)
    assert (report["attempted"], report["planned_jobs"], report["complete"]) == (1, 2, False)
    feedback = report["strategies"]["feedback"]
    assert feedback["outcomes"] == {"no_win": 1}
    assert feedback["search_threshold_candidates"] == {"1.00": 1, "1.05": 1, "1.20": 0}
    assert feedback["geometric_mean_final_speedup"] == 1.0
    assert feedback["reported_output_tokens"] == 40
    assert report["matched_feedback_independent"] == []
