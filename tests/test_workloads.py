from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hephaestus.workloads import WORKLOAD_NAMES, get_workload

EXPECTED_WORKLOADS = {
    "dynamic_batch_text",
    "graph_break_bait",
    "mlp_stack",
    "transformer_block",
}


def _eager_outputs(name: str) -> tuple[torch.Tensor, ...]:
    spec = get_workload(name)
    module = spec.make_module()
    assert not module.training
    with torch.inference_mode():
        return tuple(module(*inputs) for inputs in spec.input_cases())


def test_registry_contains_exactly_the_pinned_workloads() -> None:
    assert set(WORKLOAD_NAMES) == EXPECTED_WORKLOADS
    assert {get_workload(name).name for name in WORKLOAD_NAMES} == EXPECTED_WORKLOADS


@pytest.mark.parametrize("name", sorted(EXPECTED_WORKLOADS))
def test_seeded_construction_repeats_eager_outputs(name: str) -> None:
    first = _eager_outputs(name)
    second = _eager_outputs(name)

    assert len(first) == len(second)
    for actual, expected in zip(first, second, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", sorted(EXPECTED_WORKLOADS))
def test_workloads_run_float32_on_cpu_with_declared_tolerances(name: str) -> None:
    spec = get_workload(name)
    outputs = _eager_outputs(name)

    assert spec.dtype is torch.float32
    assert spec.atol > 0
    assert spec.rtol > 0
    for inputs in spec.input_cases():
        assert inputs
        assert all(tensor.device.type == "cpu" for tensor in inputs)
        assert all(tensor.dtype is spec.dtype for tensor in inputs)
    for output in outputs:
        assert output.device.type == "cpu"
        torch.testing.assert_close(output, output.clone(), atol=spec.atol, rtol=spec.rtol)


def test_workload_spec_is_immutable() -> None:
    spec = get_workload("mlp_stack")

    with pytest.raises(FrozenInstanceError):
        spec.seed = 999  # type: ignore[misc]


def test_constructors_isolate_only_cpu_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    real_fork_rng = torch.random.fork_rng
    observed_devices: list[list[object]] = []

    def guarded_fork_rng(*args: object, **kwargs: object) -> object:
        devices = kwargs.get("devices")
        assert devices == []
        observed_devices.append(devices)
        return real_fork_rng(*args, **kwargs)

    monkeypatch.setattr(torch.random, "fork_rng", guarded_fork_rng)

    for name in sorted(EXPECTED_WORKLOADS):
        get_workload(name).make_module()

    assert observed_devices == [[], [], [], []]


def test_dynamic_batch_text_varies_batch_and_sequence_dimensions() -> None:
    cases = get_workload("dynamic_batch_text").input_cases()
    shapes = {tuple(inputs[0].shape[:2]) for inputs in cases}

    assert len({batch for batch, _ in shapes}) > 1
    assert len({sequence for _, sequence in shapes}) > 1


def test_calibrated_mlp_stack_has_sixteen_residual_ffn_blocks() -> None:
    """Removing a residual block or changing its pinned dimensions must be detected."""
    spec = get_workload("mlp_stack")
    module = spec.make_module()
    cases = spec.input_cases()

    assert [tuple(inputs[0].shape) for inputs in cases] == [(48, 32)]
    _assert_sixteen_residual_ffn_blocks(module, cases[0][0])


def test_calibrated_transformer_pins_large_input_shape() -> None:
    """Changing the calibrated transformer input shape must fail."""
    spec = get_workload("transformer_block")
    cases = spec.input_cases()

    assert [tuple(inputs[0].shape) for inputs in cases] == [(16, 128, 32)]


def test_calibrated_transformer_retains_exact_encoder_layer_semantics() -> None:
    """Changing any constructor semantic retained during calibration must fail."""
    module = get_workload("transformer_block").make_module()

    assert isinstance(module, nn.TransformerEncoderLayer)
    assert module.self_attn.embed_dim == 32
    assert module.self_attn.num_heads == 4
    assert module.self_attn.batch_first is True
    assert module.self_attn.dropout == 0.0
    assert module.self_attn.in_proj_bias is not None
    assert module.self_attn.out_proj.bias is not None
    assert (module.linear1.in_features, module.linear1.out_features) == (32, 64)
    assert (module.linear2.in_features, module.linear2.out_features) == (64, 32)
    assert module.linear1.bias is not None
    assert module.linear2.bias is not None
    assert module.norm_first is False
    assert module.norm1.normalized_shape == module.norm2.normalized_shape == (32,)
    assert module.norm1.eps == module.norm2.eps == 1e-5
    assert module.norm1.elementwise_affine is True
    assert module.norm2.elementwise_affine is True
    assert (module.dropout.p, module.dropout1.p, module.dropout2.p) == (0.0, 0.0, 0.0)

    probe = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    torch.testing.assert_close(module.activation(probe), F.gelu(probe))


def test_calibrated_dynamic_text_has_exact_cases_and_sixteen_residual_blocks() -> None:
    """Case order, rank-three support, and all sixteen residual blocks are evidence inputs."""
    spec = get_workload("dynamic_batch_text")
    module = spec.make_module()
    cases = spec.input_cases()

    assert [tuple(inputs[0].shape) for inputs in cases] == [
        (2, 24, 32),
        (1, 48, 32),
        (3, 16, 32),
        (4, 12, 32),
    ]
    _assert_sixteen_residual_ffn_blocks(module, cases[0][0])


def test_calibrated_graph_bait_has_one_large_case_and_twelve_branch_refinements() -> None:
    """Either branch must retain its exact twelve-step data-dependent recurrence."""
    spec = get_workload("graph_break_bait")
    module = spec.make_module()
    cases = spec.input_cases()
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]

    assert len(cases) == 1
    assert tuple(cases[0][0].shape) == (128, 128)
    assert torch.equal(cases[0][0], torch.full((128, 128), 0.5))
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (128, 128)
    ]

    projection = linears[0]
    with torch.no_grad():
        projection.weight.copy_(torch.eye(128))
        projection.bias.zero_()
        positive = torch.full((2, 128), 0.5)
        negative = torch.full((2, 128), -0.5)
        expected_positive = positive
        expected_negative = negative
        for _ in range(12):
            expected_positive = torch.tanh(expected_positive * 1.01 + 0.1)
            expected_negative = torch.sin(expected_negative * 1.01 - 0.1)

        torch.testing.assert_close(module(positive), expected_positive)
        torch.testing.assert_close(module(negative), expected_negative)


def test_graph_break_bait_has_a_real_tensor_item_graph_break() -> None:
    spec = get_workload("graph_break_bait")
    module = spec.make_module()
    inputs = spec.input_cases()[0]
    torch._dynamo.reset()
    torch._dynamo.utils.graph_break_reasons.clear()

    with torch.inference_mode():
        torch.compile(module, backend="eager")(*inputs)

    reasons = [reason.reason for reason in torch._dynamo.utils.graph_break_reasons]
    assert any("Tensor.item" in reason for reason in reasons)


def _assert_sixteen_residual_ffn_blocks(
    module: nn.Module, inputs: torch.Tensor
) -> None:
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
    activations = [child for child in module.modules() if isinstance(child, nn.GELU)]
    norms = [child for child in module.modules() if isinstance(child, nn.LayerNorm)]

    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        ((32, 64) if index % 2 == 0 else (64, 32)) for index in range(32)
    ]
    assert len(activations) == 16
    assert [norm.normalized_shape for norm in norms] == [(32,)] * 16

    with torch.no_grad():
        blocks = [
            child
            for child in module.modules()
            if [type(component) for component in child.children()]
            == [nn.Linear, nn.GELU, nn.Linear, nn.LayerNorm]
        ]
        assert len(blocks) == 16

        expected = inputs.clone()
        for index, block in enumerate(blocks):
            up, _activation, down, norm = tuple(block.children())
            assert isinstance(up, nn.Linear)
            assert isinstance(down, nn.Linear)
            assert isinstance(norm, nn.LayerNorm)
            _set_deterministic_nonzero_parameters(up, down, norm, index)

            actual = block(expected)
            hidden = F.linear(expected, up.weight, up.bias)
            hidden = F.gelu(hidden)
            hidden = F.linear(hidden, down.weight, down.bias)
            expected = F.layer_norm(
                expected + hidden,
                norm.normalized_shape,
                norm.weight,
                norm.bias,
                norm.eps,
            )
            torch.testing.assert_close(actual, expected)

        torch.testing.assert_close(module(inputs), expected)


def _set_deterministic_nonzero_parameters(
    up: nn.Linear, down: nn.Linear, norm: nn.LayerNorm, block_index: int
) -> None:
    scale = float(block_index + 1)
    up.weight.copy_(
        torch.linspace(-0.02, 0.02, up.weight.numel()).reshape_as(up.weight) / scale
    )
    up.bias.copy_(torch.linspace(-0.01, 0.01, up.bias.numel()) / scale)
    down.weight.copy_(
        torch.linspace(0.015, -0.015, down.weight.numel()).reshape_as(down.weight)
        / scale
    )
    down.bias.copy_(torch.linspace(0.008, -0.008, down.bias.numel()) / scale)
    norm.weight.copy_(torch.linspace(0.9, 1.1, norm.weight.numel()))
    norm.bias.copy_(torch.linspace(-0.02, 0.02, norm.bias.numel()) / scale)
