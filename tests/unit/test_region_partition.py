import torch
import torch.nn.functional as F
from metalmind.graph.capture import capture_module
from metalmind.graph.profile import analyze_module

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(3))
        self.bias = torch.nn.Parameter(torch.zeros(3))
    def forward(self, x):
        z = x @ self.weight
        y = F.gelu(z + self.bias, approximate="tanh")
        return {"activation": y, "twice": y + y, "unchanged": torch.sin(z)}

class SharedInterior(Block):
    def forward(self, x):
        z = x + self.bias
        y = F.gelu(z, approximate="tanh")
        return y, z

class Mutating(Block):
    def forward(self, x):
        torch.manual_seed(3)
        return F.gelu(x + self.bias, approximate="tanh")

class Branch(Block):
    def forward(self, x):
        if x.sum() > 0:
            return x + self.bias
        return x


def test_partition_keeps_library_ops_and_multiple_root_consumers():
    module = Block().eval()
    captured = capture_module(module, (torch.ones(2, 3),))
    assert len(captured.regions) == 1
    region = captured.regions[0]
    assert region.info.family == "bias_gelu"
    assert set(region.info.input_names) == {"matmul", "bias"}
    torch.testing.assert_close(region.function(*region.example_inputs), captured.values[region.output])
    assert any(n.target is torch.sin for n in captured.graph.graph.nodes)
    analysis = analyze_module(module, (torch.ones(2, 3),))
    assert analysis.supported and analysis.regions[0].workload is None
    assert "MPS" in analysis.regions[0].reason
    assert analysis.whole_latency is None


def test_shared_internal_value_is_excluded():
    captured = capture_module(SharedInterior().eval(), (torch.ones(2, 3),))
    assert not captured.regions
    assert any("outside consumer" in e for e in captured.exclusions)


def test_side_effects_and_control_flow_are_rejected_before_execution():
    before = torch.random.get_rng_state().clone()
    for module in (Mutating().eval(), Branch().eval()):
        result = analyze_module(module, (torch.ones(2, 3),))
        assert not result.supported and result.reason
    assert torch.equal(before, torch.random.get_rng_state())


class OutMutation(Block):
    def forward(self, x):
        return torch.add(x, self.bias, out=x)

class AnnotatedMutation(Block):
    def forward(self, x):
        self.training: bool = True
        return x + self.bias

class CastSoftmax(Block):
    def forward(self, x):
        return torch.softmax(x.float(), dim=-1).to(x.dtype)

class TwoRegions(Block):
    def forward(self, x):
        y = F.gelu(x + self.bias, approximate="tanh")
        return torch.softmax(y, dim=-1)


def test_out_and_annotated_mutation_are_rejected_without_changing_inputs_or_state():
    for module in (OutMutation().eval(), AnnotatedMutation().eval()):
        x = torch.ones(2, 3)
        result = analyze_module(module, (x,))
        assert not result.supported
        assert not module.training
        assert torch.equal(x, torch.ones(2, 3))


def test_cast_metadata_and_dependent_regions_are_extracted():
    cast = capture_module(CastSoftmax().eval(), (torch.ones(2, 3),))
    assert len(cast.regions) == 1
    torch.testing.assert_close(cast.regions[0].function(*cast.regions[0].example_inputs), torch.softmax(torch.ones(2, 3), dim=-1))
    dependent = capture_module(TwoRegions().eval(), (torch.ones(2, 3),))
    assert [r.info.family for r in dependent.regions] == ["bias_gelu", "softmax"]
