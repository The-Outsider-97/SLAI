"""Hand-off: SoftwareAgent publishes proposals; VerificationAgent consumes them from SharedMemory."""

import uuid

import pytest

from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.software_agent import SoftwareAgent
from src.agents.verification import ProposalPolicy, VerificationStatus, check_proposal_evidence
from src.agents.verification.utils.verification_errors import MalformedSpecificationError, VerificationError
from src.agents.verification_agent import VerificationAgent, VerificationOperation

FILES = {"m.py": "VALUE = 1\n\n\ndef f():\n    return VALUE\n"}
REGRESSING_DIFF = (
    "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1,5 +1,5 @@\n"
    "-VALUE = 1\n+OTHER = 1\n \n \n def f():\n     return VALUE\n"
)


def agents(**verification_config):
    memory = SharedMemory()
    factory = AgentFactory()
    return SoftwareAgent(shared_memory=memory, agent_factory=factory), VerificationAgent(shared_memory=memory, agent_factory=factory, config=verification_config), memory


def propose(software, operation="rename", parameters=None, **payload):
    parameters = parameters if parameters is not None else {"symbol": "m.f", "new_name": "g"}
    return software.perform_task({"operation": operation, "files": FILES, "parameters": parameters, **payload})["patch_id"]


def verify(verifier, artifact, **payload):
    return verifier.perform_task({"operation": "proposal_evidence", "artifact": artifact, "request_id": uuid.uuid4().hex, **payload})


def test_published_proposal_is_verified_without_glue_code():
    software, verifier, memory = agents()
    patch_id = propose(software)
    outcome = verify(verifier, patch_id, request_id="pe-ok")

    assert outcome["execution_success"] is True and outcome["verification_status"] == "verified"
    formal = outcome["formal_result"]
    assert formal["provenance"]["method"] == "contract" and formal["provenance"]["backend"] == "recorded-evidence"
    assert set(formal["model"]) == {
        "not_applied", "gates_declared", "syntax_valid", "no_introduced_regressions",
        "bounded_unresolved_sites", "confidence_floor", "identity_consistent",
    }
    assert all(text.startswith("holds") for text in formal["model"].values())
    assert any("behavioural equivalence" in item for item in formal["limitations"])

    verdict = memory.get(f"verification_agent.proposal_verdict:{patch_id}")
    assert verdict["verification_status"] == "verified" and verdict["violated"] == [] and verdict["behaviour_verified"] is False
    assert memory.get("verification_agent.summary:pe-ok")["verification_status"] == "verified"
    assert memory.get("verification_agent.request:pe-ok")["patch_id"] == patch_id


def test_new_correctness_findings_refute_the_proposal():
    software, verifier, memory = agents()
    patch_id = propose(software, "apply_diff", {"diff": REGRESSING_DIFF})
    outcome = verify(verifier, patch_id)
    assert outcome["verification_status"] == "refuted"
    assert outcome["formal_result"]["model"]["no_introduced_regressions"].startswith("violated")
    verdict = memory.get(f"verification_agent.proposal_verdict:{patch_id}")
    assert verdict["violated"] == ["no_introduced_regressions"]


def test_policy_comes_from_options_or_an_explicit_policy_object():
    software, verifier, _ = agents()
    patch_id = propose(software)
    strict = verify(verifier, patch_id, options={"min_confidence": 0.99})
    assert strict["verification_status"] == "refuted" and strict["formal_result"]["model"]["confidence_floor"].startswith("violated")

    regressing = propose(software, "apply_diff", {"diff": REGRESSING_DIFF})
    lenient = verify(verifier, regressing, options={"max_introduced_regressions": 5})
    assert lenient["verification_status"] == "verified"
    explicit = verifier.verify({"operation": "proposal_evidence", "artifact": regressing, "formal_property": ProposalPolicy(max_introduced_regressions=5)})
    assert explicit.verification_status is VerificationStatus.VERIFIED


def test_missing_evidence_is_unknown_and_tampered_evidence_is_refuted():
    base = {
        "patch_id": "p", "base_snapshot_digest": "a", "result_snapshot_digest": "b", "applied": False,
        "requires_gates": ["verification", "safety", "execution", "evaluation"], "unresolved_sites": 0, "confidence": 0.9,
        "validation": {"syntax": {"ok": True}, "static_regressions": {"introduced_count": 0}},
    }
    assert check_proposal_evidence(base).status is VerificationStatus.VERIFIED

    incomplete = {key: value for key, value in base.items() if key != "validation"}
    result = check_proposal_evidence(incomplete)
    assert result.status is VerificationStatus.UNKNOWN and "syntax_valid" in result.unknown_reason

    assert check_proposal_evidence(dict(base, applied=True)).status is VerificationStatus.REFUTED
    assert check_proposal_evidence(dict(base, requires_gates=["verification"])).status is VerificationStatus.REFUTED
    assert check_proposal_evidence(dict(base, result_snapshot_digest="a")).status is VerificationStatus.REFUTED
    # A violation outranks missing evidence.
    assert check_proposal_evidence(dict(incomplete, applied=True)).status is VerificationStatus.REFUTED


def test_inline_evidence_is_accepted_and_bad_inputs_are_operational_errors():
    software, verifier, memory = agents()
    patch_id = propose(software)
    record = memory.get(f"software_agent.proposal:{patch_id}")
    assert verify(verifier, dict(record))["verification_status"] == "verified"

    with pytest.raises(MalformedSpecificationError):
        verify(verifier, "patch-never-published")
    with pytest.raises(MalformedSpecificationError):
        verify(verifier, 42)
    with pytest.raises(MalformedSpecificationError):
        verify(verifier, patch_id, options={"min_confidence": 7})
    with pytest.raises(VerificationError):
        ProposalPolicy.from_mapping({"nonsense": 1})

    other = propose(software, parameters={"symbol": "m.f", "new_name": "h"})
    memory.set(f"software_agent.proposal:{patch_id}", memory.get(f"software_agent.proposal:{other}"))
    with pytest.raises(MalformedSpecificationError):
        verify(verifier, patch_id)

    offline = VerificationAgent(shared_memory=memory, agent_factory=AgentFactory(), config={"publish_shared_memory": False})
    with pytest.raises(MalformedSpecificationError):
        verify(offline, other)


def test_operation_is_advertised_and_existing_operations_are_untouched():
    _, verifier, _ = agents()
    assert "proposal_evidence" in verifier.capabilities()["operations"]
    assert VerificationOperation("proposal_evidence") is VerificationOperation.PROPOSAL_EVIDENCE
    assert len(VerificationOperation) == 8
