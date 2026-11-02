from metalmind.artifacts.loader import load_artifact


def test_missing_or_corrupt_artifact_returns_original_function(tmp_path):
    fn=load_artifact(tmp_path,fallback=lambda x:x+2)
    assert fn(3)==5
    assert fn.reason
