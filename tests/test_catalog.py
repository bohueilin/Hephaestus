from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from hephaestus.catalog import (
    CATALOG,
    CATALOG_BY_ID,
    BatchBucketPolicy,
    CatalogEntry,
    CatalogRole,
    CompileConfig,
    CompileMode,
    DynamicStrategy,
    FullGraph,
    OptionSetting,
    Proposal,
    ReadOnlyRunResult,
    WorkloadName,
    catalog_entries_for_optimizer,
    catalog_json,
    get_catalog_entry,
    resolve_action,
    validate_catalog_capabilities,
)


def _config(**overrides: object) -> CompileConfig:
    values: dict[str, object] = {
        "mode": CompileMode.DEFAULT,
        "dynamic": DynamicStrategy.FALSE,
        "fullgraph": FullGraph.TRUE,
        "options": (),
        "disable": False,
        "bucket_policy": None,
    }
    values.update(overrides)
    return CompileConfig(**values)  # type: ignore[arg-type]


def _snapshot(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": 1,
        "torch_version": "2.13.0",
        "compiler_backends": ["inductor"],
        "inductor_modes": [
            "default",
            "max-autotune-no-cudagraphs",
            "reduce-overhead",
        ],
        "inductor_options": ["epilogue_fusion"],
    }
    values.update(overrides)
    return values


def test_catalog_has_exactly_three_plants_and_one_clean_control() -> None:
    """Adding or dropping a regression must not silently change the calibrated demo set."""
    planted = tuple(entry for entry in CATALOG if entry.role is CatalogRole.PLANTED)
    controls = tuple(entry for entry in CATALOG if entry.role is CatalogRole.CLEAN_CONTROL)

    assert [entry.catalog_id for entry in planted] == [
        "planted-eager-fallback",
        "planted-static-shape-storm",
        "planted-graph-break-exposure",
    ]
    assert [entry.catalog_id for entry in controls] == ["clean-control-mlp"]
    assert [(entry.expected_verdict, entry.expected_driving_finding) for entry in planted] == [
        ("NOT_PROVEN", "perf.speedup_proven"),
        ("NOT_PROVEN", "graph.recompile_bound"),
        ("CONDITIONAL", "graph.no_breaks"),
    ]
    assert (controls[0].expected_verdict, controls[0].expected_driving_finding) == (
        "PROVEN",
        "all_criteria_passed",
    )


def test_optimizer_catalog_covers_every_workload_without_demo_entries() -> None:
    """A search must have an action for each workload but never execute demo authority."""
    by_workload = {
        workload: catalog_entries_for_optimizer(workload) for workload in WorkloadName
    }

    assert all(by_workload.values())
    assert all(
        entry.role is CatalogRole.CANDIDATE
        for entries in by_workload.values()
        for entry in entries
    )
    assert [entry.catalog_id for entry in by_workload[WorkloadName.DYNAMIC_BATCH_TEXT]][:3] == [
        "candidate-dynamic-static",
        "candidate-dynamic-true",
        "candidate-dynamic-bucketed",
    ]


def test_catalog_and_nested_settings_are_deeply_immutable() -> None:
    """Callers must not mutate a named action after it has passed catalog review."""
    entry = get_catalog_entry("candidate-mlp-epilogue-fusion-off")

    with pytest.raises(FrozenInstanceError):
        entry.config.disable = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        CATALOG_BY_ID[entry.catalog_id] = entry  # type: ignore[index]
    assert isinstance(entry.config.options, tuple)
    assert entry.config.options == (OptionSetting("epilogue_fusion", False),)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("mode", "default"),
        ("dynamic", "false"),
        ("fullgraph", "true"),
        ("options", [OptionSetting("epilogue_fusion", False)]),
        ("disable", 1),
    ],
)
def test_compile_config_rejects_values_outside_typed_schema(
    field: str, bad_value: object
) -> None:
    """Untyped or mutable settings must fail before they can reach Torch."""
    with pytest.raises(ValueError):
        _config(**{field: bad_value})


def test_options_reject_duplicates_nonscalar_values_and_mode_conflicts() -> None:
    """An option cannot be ambiguous, mutable, or combined with a conflicting Torch mode."""
    with pytest.raises(ValueError, match="duplicate"):
        _config(
            options=(
                OptionSetting("epilogue_fusion", False),
                OptionSetting("epilogue_fusion", True),
            )
        )
    with pytest.raises(ValueError, match="scalar"):
        OptionSetting("epilogue_fusion", [False])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        _config(
            mode=CompileMode.REDUCE_OVERHEAD,
            options=(OptionSetting("epilogue_fusion", False),),
        )


def test_bucket_policy_is_frozen_and_legal_only_for_dynamic_text() -> None:
    """A proposal cannot invent buckets or apply padding semantics to another workload."""
    policy = BatchBucketPolicy(boundaries=(2, 4))
    bucketed = _config(dynamic=DynamicStrategy.BUCKETED, bucket_policy=policy)

    with pytest.raises(ValueError, match="exactly"):
        BatchBucketPolicy(boundaries=(2, 8))
    with pytest.raises(ValueError, match="bucket"):
        _config(dynamic=DynamicStrategy.BUCKETED)
    with pytest.raises(ValueError, match="bucket"):
        _config(bucket_policy=policy)
    with pytest.raises(ValueError, match="dynamic_batch_text"):
        CatalogEntry(
            catalog_id="candidate-illegal-bucket",
            workload=WorkloadName.MLP_STACK,
            config=bucketed,
            role=CatalogRole.CANDIDATE,
            rationale="Use frozen buckets.",
        )


def test_disable_is_reserved_for_the_named_planted_fallback() -> None:
    """No normal candidate can turn compilation off under a plausible-looking ID."""
    with pytest.raises(ValueError, match="eager fallback"):
        CatalogEntry(
            catalog_id="candidate-disabled",
            workload=WorkloadName.MLP_STACK,
            config=_config(disable=True),
            role=CatalogRole.CANDIDATE,
            rationale="Try a disabled compiler.",
        )


def test_proposals_validate_closed_ids_workloads_and_one_line_rationales() -> None:
    """Unknown actions, mismatched workloads, and multiline text must fail at the seam."""
    with pytest.raises(ValueError, match="unknown catalog"):
        Proposal("not-in-catalog", WorkloadName.MLP_STACK, "Try it.")
    with pytest.raises(ValueError, match="workload"):
        Proposal(
            "candidate-mlp-default",
            WorkloadName.TRANSFORMER_BLOCK,
            "Try the default.",
        )
    with pytest.raises(ValueError, match="one-line"):
        Proposal("candidate-mlp-default", WorkloadName.MLP_STACK, "first\nsecond")
    with pytest.raises(ValueError, match="nonempty"):
        Proposal("candidate-mlp-default", WorkloadName.MLP_STACK, "   ")


def test_unknown_lookup_and_string_workload_are_rejected() -> None:
    """Stringly typed or unknown catalog references must not become executable actions."""
    with pytest.raises(ValueError, match="unknown catalog"):
        get_catalog_entry("forged")
    with pytest.raises(ValueError, match="workload"):
        catalog_entries_for_optimizer("mlp_stack")  # type: ignore[arg-type]


def test_resolver_maps_requested_settings_to_exact_effective_torch_arguments() -> None:
    """Changing a mode, dynamic, fullgraph, option, or disable mapping must be detected."""
    cases = {
        "candidate-mlp-default": (None, False, True, None, False, "static"),
        "candidate-mlp-reduce-overhead": (
            "reduce-overhead",
            False,
            True,
            None,
            False,
            "static",
        ),
        "candidate-mlp-max-autotune-equivalent": (
            "max-autotune-no-cudagraphs",
            False,
            True,
            None,
            False,
            "static",
        ),
        "candidate-mlp-epilogue-fusion-off": (
            None,
            False,
            True,
            {"epilogue_fusion": False},
            False,
            "static",
        ),
        "candidate-dynamic-true": (None, True, True, None, False, "dynamic"),
        "candidate-dynamic-bucketed": (None, False, True, None, False, "bucketed"),
        "planted-eager-fallback": (None, False, True, None, True, "static"),
    }

    for catalog_id, expected in cases.items():
        action = resolve_action(get_catalog_entry(catalog_id))
        observed = (
            action.mode,
            action.dynamic,
            action.fullgraph,
            None if action.options is None else dict(action.options),
            action.disable,
            action.input_plan_strategy,
        )
        assert observed == expected
        assert action.backend == "inductor"


def test_capability_snapshot_validation_does_not_grant_arbitrary_options() -> None:
    """Missing references must fail, while discovery cannot create an executable ID."""
    validate_catalog_capabilities(_snapshot(inductor_options=["epilogue_fusion", "debug"]))

    with pytest.raises(ValueError, match="backend"):
        validate_catalog_capabilities(_snapshot(compiler_backends=["eager"]))
    with pytest.raises(ValueError, match="mode"):
        validate_catalog_capabilities(_snapshot(inductor_modes=["default"]))
    with pytest.raises(ValueError, match="option"):
        validate_catalog_capabilities(_snapshot(inductor_options=["debug"]))
    with pytest.raises(ValueError, match="unknown catalog"):
        get_catalog_entry("candidate-debug")


def test_parent_catalog_evidence_preserves_rationales_and_demo_expectations() -> None:
    """Removing reviewed text or expected outcomes must not leave an incomplete catalog trace."""
    payload = catalog_json("0" * 64)
    entries = {entry["entry_id"]: entry for entry in payload["entries"]}  # type: ignore[index]

    planted = entries["planted-eager-fallback"]
    assert planted["rationale"] == "Plant a compiler-disabled eager fallback."
    assert planted["expected_verdict"] == "NOT_PROVEN"
    assert planted["expected_driving_finding"] == "perf.speedup_proven"
    candidate = entries["candidate-mlp-default"]
    assert candidate["expected_verdict"] is None
    assert candidate["expected_driving_finding"] is None
    assert payload["torch_capabilities_sha256"] == "0" * 64


@pytest.mark.parametrize(
    "path",
    ["", ".", "/absolute", "runs/../escape", "runs\\child", "runs//child"],
)
def test_read_only_results_reject_unsafe_or_noncanonical_paths(path: str) -> None:
    """An immutable receipt path cannot escape or ambiguously address the search tree."""
    with pytest.raises(ValueError, match="path"):
        ReadOnlyRunResult(path, "PROVEN", "all_criteria_passed")


def test_read_only_result_has_only_three_frozen_scalar_fields() -> None:
    """Adding raw evidence or mutable state would expand the agent's observation authority."""
    result = ReadOnlyRunResult("runs/child", "NOT_PROVEN", "perf.speedup_proven")

    assert result.bundle_relative_path == "runs/child"
    assert result.verdict == "NOT_PROVEN"
    assert result.driving_finding == "perf.speedup_proven"
    assert tuple(result.__dataclass_fields__) == (
        "bundle_relative_path",
        "verdict",
        "driving_finding",
    )
    with pytest.raises(FrozenInstanceError):
        result.verdict = "PROVEN"  # type: ignore[misc]
