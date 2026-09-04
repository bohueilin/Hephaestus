"""Inert scripted policy: proposals in, scalar observations back, no execution authority."""

from __future__ import annotations

from dataclasses import dataclass

from hephaestus.catalog import Proposal, WorkloadName, catalog_entries_for_optimizer


@dataclass(frozen=True, slots=True)
class AgentObservation:
    """The complete scalar result visible to the policy after one trusted run."""

    verdict: str
    driving_finding: str

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, str) or not self.verdict:
            raise ValueError("observation verdict must be nonempty")
        if not isinstance(self.driving_finding, str) or not self.driving_finding:
            raise ValueError("observation finding must be nonempty")


@dataclass(frozen=True, slots=True)
class ScriptedPolicy:
    """Closed ordered proposal data for one workload."""

    workload_name: WorkloadName
    proposals: tuple[Proposal, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.workload_name, WorkloadName):
            raise ValueError("policy workload must be a WorkloadName")
        if not isinstance(self.proposals, tuple) or any(
            type(proposal) is not Proposal for proposal in self.proposals
        ):
            raise ValueError("policy proposals must be an immutable Proposal tuple")
        if any(
            proposal.workload_name is not self.workload_name
            for proposal in self.proposals
        ):
            raise ValueError("policy proposal workload mismatch")


def scripted_policy(workload_name: WorkloadName) -> ScriptedPolicy:
    """Construct the closed v0.1 candidate sequence without an execution callback."""
    if not isinstance(workload_name, WorkloadName):
        raise ValueError("workload_name must be a WorkloadName")
    proposals = tuple(
        Proposal(entry.catalog_id, workload_name, entry.rationale)
        for entry in catalog_entries_for_optimizer(workload_name)
    )
    return ScriptedPolicy(workload_name, proposals)


def next_proposal(
    policy: ScriptedPolicy,
    observations: tuple[AgentObservation, ...],
) -> Proposal | None:
    """Return the next fixed proposal, stopping only after PROVEN or exhaustion."""
    if not isinstance(policy, ScriptedPolicy):
        raise ValueError("policy must be a ScriptedPolicy")
    if not isinstance(observations, tuple) or any(
        not isinstance(observation, AgentObservation) for observation in observations
    ):
        raise ValueError("observations must be an immutable scalar observation tuple")
    if len(observations) > len(policy.proposals):
        raise ValueError("observations exceed the closed proposal sequence")
    if any(observation.verdict == "PROVEN" for observation in observations):
        return None
    if len(observations) == len(policy.proposals):
        return None
    return policy.proposals[len(observations)]


__all__ = ["AgentObservation", "ScriptedPolicy", "next_proposal", "scripted_policy"]
