from pathlib import Path
import subprocess
import pytest
from metalmind.demonstration.provenance import package_fingerprint, git_package_fingerprint, validate_runtime_source


def test_runtime_source_must_match_committed_code_not_dirty_checkout(tmp_path):
    root=tmp_path/"repo";package=root/"src/metalmind";package.mkdir(parents=True)
    file=package/"__init__.py";file.write_text("VALUE = 1\n")
    subprocess.run(["git","init","-q"],cwd=root,check=True)
    subprocess.run(["git","add","."],cwd=root,check=True)
    subprocess.run(["git","-c","user.name=Test","-c","user.email=test@example.invalid","commit","-qm","fixture"],cwd=root,check=True)
    expected=git_package_fingerprint(root)
    assert package_fingerprint(package)==expected
    validate_runtime_source(expected,expected)
    file.write_text("VALUE = 2\n")
    actual=package_fingerprint(package)
    assert actual!=expected
    with pytest.raises(ValueError,match="package|source"): validate_runtime_source(actual,expected)
