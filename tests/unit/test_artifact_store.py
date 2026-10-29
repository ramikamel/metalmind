import pytest
from metalmind.artifacts.store import ArtifactStore


def test_immutable_candidates_and_truncated_event_recovery(tmp_path):
    store=ArtifactStore(tmp_path)
    store.write_source("abc","first")
    with pytest.raises(ValueError): store.write_source("abc","changed")
    store.append_event({"event":"created"})
    with (tmp_path/"events.jsonl").open("a") as f: f.write('{"event":')
    assert store.read_events()==[{"event":"created"}]
    store.append_event({"event":"resumed"})
    assert [e['event'] for e in store.read_events()]==['created','recovered_truncated_tail','resumed']


def test_manifest_replacement_preserves_complete_json(tmp_path):
    store=ArtifactStore(tmp_path);store.write_manifest({"phase":"first"})
    store.write_manifest({"phase":"second"})
    assert store.read_manifest()=={"phase":"second"}
    with pytest.raises(ValueError): store.write_source("../escape","bad")
