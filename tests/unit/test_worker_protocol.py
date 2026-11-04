from contextlib import contextmanager
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from metalmind.contracts import EvaluationRequest, CandidateSpec, CandidateProposal
from metalmind.runtime import lock, worker
from metalmind.runtime.abi import signature_for


@pytest.fixture
def evaluation_request(sample_workload):
    candidate=CandidateSpec(candidate_id="cancel",workload_hash=signature_for(sample_workload),
        proposal=CandidateProposal(source="invalid",threads=(6,1,1),group_size=(2,1,1),hypothesis="test"))
    return EvaluationRequest(candidate=candidate,workload=sample_workload)


def test_already_cancelled_evaluation_does_not_spawn(evaluation_request):
    cancelled = threading.Event()
    cancelled.set()
    assert worker.evaluate(evaluation_request, cancel_event=cancelled).status == "cancelled"


@pytest.fixture
def worker_process(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "FAULT_PATH", tmp_path / "fault.json")
    process = Mock(pid=None)
    process.start.side_effect = lambda: setattr(process, "pid", 123)
    process.is_alive.side_effect = lambda: process.pid is not None
    process.kill.side_effect = lambda: setattr(process, "pid", None)
    receive = Mock()
    receive.poll.return_value = True
    receive.recv.side_effect = EOFError
    context = SimpleNamespace(Pipe=lambda **kwargs: (receive, Mock()), Process=lambda **kwargs: process)
    monkeypatch.setattr(worker.mp, "get_context", lambda method: context)
    return process


@pytest.mark.parametrize(("state", "status"), [("fault", "runtime_error"), ("cancelled", "cancelled")])
def test_waiting_evaluator_rechecks_state_under_lock(
    evaluation_request, worker_process, monkeypatch, state, status,
):
    cancelled = threading.Event()

    @contextmanager
    def gpu_lock(timeout):
        if state == "fault":
            worker._record_fault("Previous evaluation crashed")
        else:
            cancelled.set()
        yield

    monkeypatch.setattr(lock, "gpu_lock", gpu_lock)

    result = worker.evaluate(evaluation_request, cancel_event=cancelled)

    assert result.status == status
    assert result.device_fault is (state == "fault")
    worker_process.start.assert_not_called()


def test_crashed_evaluator_reaps_child_before_unlocking(
    evaluation_request, worker_process, monkeypatch,
):
    children_alive_at_unlock = []

    @contextmanager
    def gpu_lock(timeout):
        try:
            yield
        finally:
            children_alive_at_unlock.append(worker_process.is_alive())

    monkeypatch.setattr(lock, "gpu_lock", gpu_lock)

    result = worker.evaluate(evaluation_request)

    assert result.status == "runtime_error"
    assert result.device_fault and worker.gpu_faulted()
    assert children_alive_at_unlock == [False]


@pytest.mark.parametrize("late_stage_message", [False, True])
def test_poll_cannot_accept_a_message_after_the_compile_deadline(
    evaluation_request, worker_process, monkeypatch, late_stage_message,
):
    from contextlib import nullcontext

    now = [0.0]
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(lock, "gpu_lock", lambda timeout: nullcontext())
    receive, _ = worker.mp.get_context("spawn").Pipe()
    reply = {"result": {"status": "compile_error"}}

    def delayed_poll(timeout):
        now[0] = 1.1
        return True

    receive.poll.side_effect = delayed_poll
    receive.recv.side_effect = [{"stage": "execution"}, reply] if late_stage_message else [reply]
    request = evaluation_request.model_copy(update={
        "compile_timeout_seconds": 1.0, "execution_timeout_seconds": 8.0, "total_timeout_seconds": 10.0,
    })
    result = worker.evaluate(request)
    assert result.status == "timeout" and result.device_fault
