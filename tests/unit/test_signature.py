from metalmind.runtime.abi import signature_for, matches_signature
import torch


def test_identity_tracks_semantics_and_layout(sample_workload):
    before=signature_for(sample_workload)
    for name,value in (("epsilon",1e-4),("accumulation_dtype","float64")):
        changed=sample_workload.model_copy(update={name:value})
        assert signature_for(changed) != before
    changed=sample_workload.model_copy(update={"inputs": (
        sample_workload.inputs[0].model_copy(update={"storage_offset":1}),sample_workload.inputs[1])})
    assert signature_for(changed) != before


def test_cpu_inputs_do_not_match_mps_signature(sample_workload):
    assert not matches_signature(sample_workload, (torch.zeros(2,3),torch.zeros(3)))
