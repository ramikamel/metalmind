from pathlib import Path
import pytest
from metalmind.verification.final_gate import run_final_gate


def test_consumed_private_gate_cannot_be_reused(tmp_path):
    marker=tmp_path/"gate";marker.write_text("already used")
    with pytest.raises(FileExistsError):
        run_final_gate(None,None,(),marker_path=marker)
