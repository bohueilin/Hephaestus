from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path

import pytest

from hephaestus import agent as agent_module
from hephaestus.agent import AgentObservation, ScriptedPolicy, next_proposal, scripted_policy
from hephaestus.catalog import Proposal, WorkloadName


def test_policy_uses_fixed_order_and_stops_only_on_proven() -> None:
    policy = scripted_policy(WorkloadName.DYNAMIC_BATCH_TEXT)
    observations: tuple[AgentObservation, ...] = ()

    first = next_proposal(policy, observations)
    assert first is not None and first.catalog_id == "candidate-dynamic-static"
    observations += (AgentObservation("NOT_PROVEN", "graph.recompile_bound"),)
    second = next_proposal(policy, observations)
    assert second is not None and second.catalog_id == "candidate-dynamic-true"
    observations += (AgentObservation("PROVEN", "all_criteria_passed"),)
    assert next_proposal(policy, observations) is None


@pytest.mark.parametrize("nonterminal", ["CONDITIONAL", "NOT_PROVEN", "INVALID_EVIDENCE"])
def test_only_proven_is_terminal(nonterminal: str) -> None:
    policy = scripted_policy(WorkloadName.DYNAMIC_BATCH_TEXT)
    proposal = next_proposal(
        policy,
        (AgentObservation(nonterminal, "first.finding"),),
    )
    assert proposal is not None and proposal.catalog_id == "candidate-dynamic-true"


def test_exhausted_policy_returns_none_without_fabricating_success() -> None:
    policy = scripted_policy(WorkloadName.GRAPH_BREAK_BAIT)
    assert next_proposal(
        policy,
        (AgentObservation("CONDITIONAL", "graph.no_breaks"),),
    ) is None


def test_policy_rejects_unknown_workload() -> None:
    with pytest.raises(ValueError, match="workload"):
        scripted_policy("forged")  # type: ignore[arg-type]


def test_policy_never_proposes_planted_or_control_entries() -> None:
    for workload in WorkloadName:
        policy = scripted_policy(workload)
        assert policy.proposals
        assert all(
            not proposal.catalog_id.startswith(("planted-", "clean-control-"))
            for proposal in policy.proposals
        )


def test_policy_data_is_immutable_and_path_free() -> None:
    policy = scripted_policy(WorkloadName.MLP_STACK)
    with pytest.raises(FrozenInstanceError):
        policy.workload_name = WorkloadName.GRAPH_BREAK_BAIT  # type: ignore[misc]
    assert tuple(field.name for field in fields(AgentObservation)) == (
        "verdict",
        "driving_finding",
    )
    assert all(not hasattr(proposal, "bundle_path") for proposal in policy.proposals)


def test_agent_is_an_inert_policy_with_scalar_observations_and_no_execution_callback() -> None:
    observation_type = getattr(agent_module, "AgentObservation", None)
    policy_builder = getattr(agent_module, "scripted_policy", None)
    transition = getattr(agent_module, "next_proposal", None)

    assert observation_type is not None
    assert callable(policy_builder)
    assert callable(transition)
    policy = policy_builder(WorkloadName.DYNAMIC_BATCH_TEXT)
    first = transition(policy, ())
    assert first.catalog_id == "candidate-dynamic-static"
    refusal = observation_type("NOT_PROVEN", "graph.recompile_bound")
    second = transition(policy, (refusal,))
    assert second.catalog_id == "candidate-dynamic-true"
    proven = observation_type("PROVEN", "all_criteria_passed")
    assert transition(policy, (refusal, proven)) is None
    assert tuple(observation_type.__dataclass_fields__) == ("verdict", "driving_finding")
    assert not hasattr(agent_module, "ScriptedOptimizer")


def test_agent_source_has_no_ambient_execution_modules_or_callable_dtos() -> None:
    source = inspect.getsource(agent_module)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    forbidden_prefixes = (
        "pathlib",
        "os",
        "subprocess",
        "socket",
        "http",
        "urllib",
        "hephaestus.gate",
        "hephaestus.catalog_runtime",
        "hephaestus.measure",
        "hephaestus.bundle",
    )
    assert not any(
        imported_name == prefix or imported_name.startswith(f"{prefix}.")
        for imported_name in imported
        for prefix in forbidden_prefixes
    )
    assert "ResultsAPI" not in source
    assert "Callable" not in source
    assert "HostState" not in source
    assert "host_state" not in source
    assert "sink" not in source


def test_policy_constructor_rejects_ambient_paths_and_callable_proposals() -> None:
    """Runtime construction cannot smuggle filesystem or execution authority into policy DTOs."""
    authentic = scripted_policy(WorkloadName.MLP_STACK)

    with pytest.raises(ValueError, match="workload"):
        ScriptedPolicy(Path("/private/ambient"), authentic.proposals)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="proposal"):
        ScriptedPolicy(WorkloadName.MLP_STACK, (lambda: None,))  # type: ignore[arg-type]


def test_policy_constructor_rejects_proposal_subclasses_with_ambient_authority() -> None:
    """A Proposal subclass cannot add callable authority to the closed policy DTO."""

    @dataclass(frozen=True, slots=True)
    class AmbientProposal(Proposal):
        execute: object

    authentic = scripted_policy(WorkloadName.MLP_STACK).proposals[0]
    ambient = AmbientProposal(
        authentic.catalog_id,
        authentic.workload_name,
        authentic.rationale,
        lambda: None,
    )

    with pytest.raises(ValueError, match="proposal"):
        ScriptedPolicy(WorkloadName.MLP_STACK, (ambient,))
