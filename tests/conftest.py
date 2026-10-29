import pytest
from metalmind.contracts import WorkloadSpec, TensorSpec

@pytest.fixture
def sample_workload():
    return WorkloadSpec(workload_id="bias_gelu:tiny", family="bias_gelu",
        inputs=(TensorSpec(shape=(2, 3), strides=(3, 1)), TensorSpec(shape=(3,), strides=(1,))),
        output=TensorSpec(shape=(2, 3), strides=(3, 1)))
