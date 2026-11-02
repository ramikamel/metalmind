import pytest
from metalmind.evaluation import CampaignConfig, freeze_campaign


def test_campaign_identity_tracks_budget_and_domain_mode():
    a=CampaignConfig();b=CampaignConfig(domain_mode="assume")
    assert freeze_campaign(a)['campaign_id']!=freeze_campaign(b)['campaign_id']
    assert freeze_campaign(a)['jobs']==108  # 3 families, 3 shapes, 3 trials, 4 strategies


def test_campaign_identity_binds_actual_workloads_and_tolerance(monkeypatch):
    import metalmind.evaluation as evaluation
    from metalmind.verification.policy import policy_for
    config = CampaignConfig(families=("rmsnorm",), cases=("small",))
    original = freeze_campaign(config)
    real_get = evaluation.get_workload
    monkeypatch.setattr(evaluation, "get_workload", lambda *a: real_get(*a).model_copy(update={"epsilon": 1e-4}))
    assert freeze_campaign(config)["campaign_id"] != original["campaign_id"]
    monkeypatch.setattr(evaluation, "get_workload", real_get)
    monkeypatch.setattr(evaluation, "policy_for", lambda w: policy_for(w).model_copy(update={"atol": 1e-3}))
    assert freeze_campaign(config)["campaign_id"] != original["campaign_id"]
    assert original["workloads"][0]["workload"]["reference_hash"]
