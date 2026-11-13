"""Project-scoped, restart-safe coordination tests; no external host is mocked as a commit."""

import threading

import pytest

from src.agents.collaborative.project_coordination import ProjectCoordinationError, ProjectCoordinationStore
from src.agents.collaborative_agent import CollaborativeAgent
from src.agents.planning.workflow_runtime import AuthorizationContext, WorkflowAuthorizationError


def auth(actor="alice", project="p1", *scopes):
    default = scopes or ("collaboration:join", "collaboration:read", "collaboration:lock")
    return AuthorizationContext(actor, project, frozenset(default))


def test_project_participants_and_authorization_isolation(tmp_path):
    store = ProjectCoordinationStore(tmp_path / "coordination.db")
    alice, bob, other = auth(), auth("bob"), auth("other", "p2")
    store.join(auth=alice, kind="human")
    assert store.join(auth=alice, kind="human")["active"]
    store.join(auth=bob, kind="agent")
    store.join(auth=other, kind="service")
    assert store.participants(auth=alice) == [
        {"actor_id": "alice", "kind": "human"}, {"actor_id": "bob", "kind": "agent"}]
    assert store.participants(auth=other) == [{"actor_id": "other", "kind": "service"}]
    assert [event["event_type"] for event in store.events(auth=alice)] == [
        "participant.joined", "participant.joined"]
    with pytest.raises(WorkflowAuthorizationError):
        store.acquire("r1", auth=auth("alice", "p1", "collaboration:read"))
    with pytest.raises(ProjectCoordinationError, match="join"):
        store.acquire("r1", auth=auth("intruder"))
    store.leave(auth=bob)
    assert store.participants(auth=alice) == [{"actor_id": "alice", "kind": "human"}]
    with pytest.raises(ProjectCoordinationError, match="join"):
        store.events(auth=bob)
    store.close()


def test_fencing_expiry_renewal_recovery_and_old_owner_rejection(tmp_path):
    now = [1000.0]
    path = tmp_path / "coordination.db"
    first = ProjectCoordinationStore(path, clock=lambda: now[0])
    alice, bob = auth(), auth("bob")
    first.join(auth=alice, kind="human")
    first.join(auth=bob, kind="agent")
    lease1 = first.acquire("document", auth=alice, ttl=10)
    assert lease1["fencing_token"] == 1
    with pytest.raises(ProjectCoordinationError, match="active"):
        first.acquire("document", auth=bob)
    with pytest.raises(ProjectCoordinationError, match="not owned"):
        first.release("document", lease1["lease_id"], lease1["fencing_token"], auth=bob)
    renewed = first.renew("document", lease1["lease_id"], lease1["fencing_token"], auth=alice, ttl=15)
    assert renewed["expires_at"] == 1015.0
    first.close()

    recovered = ProjectCoordinationStore(path, clock=lambda: now[0])
    assert recovered.check_lease("document", lease1["lease_id"], 1, auth=alice)
    now[0] = 1016.0
    assert not recovered.check_lease("document", lease1["lease_id"], 1, auth=alice)
    lease2 = recovered.acquire("document", auth=bob)
    assert lease2["fencing_token"] == 2
    with pytest.raises(ProjectCoordinationError, match="expired, invalid"):
        recovered.renew("document", lease1["lease_id"], 1, auth=alice)
    with pytest.raises(ProjectCoordinationError, match="expired, invalid"):
        recovered.release("document", lease1["lease_id"], 1, auth=alice)
    recovered.release("document", lease2["lease_id"], 2, auth=bob)
    lease3 = recovered.acquire("document", auth=alice)
    assert lease3["fencing_token"] == 3
    recovered.close()


def test_events_replay_and_project_partition(tmp_path):
    store = ProjectCoordinationStore(tmp_path / "coordination.db")
    a, b = auth(), auth("bob", "p2")
    store.join(auth=a, kind="human")
    cursor = store.events(auth=a)[-1]["sequence"]
    store.join(auth=b, kind="agent")
    lock = store.acquire("resource", auth=a)
    snapshot = store.snapshot(auth=a)
    assert snapshot["leases"][0]["token"] == lock["fencing_token"]
    assert snapshot["leases"][0]["resource_id"] == "resource"
    assert store.events(auth=a, after=snapshot["event_cursor"]) == []
    assert [e["event_type"] for e in store.events(auth=a, after=cursor)] == ["lease.acquired"]
    assert [e["event_type"] for e in store.events(auth=b)] == ["participant.joined"]
    store.release("resource", lock["lease_id"], lock["fencing_token"], auth=a)
    assert store.events(auth=a, after=cursor)[-1]["event_type"] == "lease.released"
    store.close()


def test_parallel_resource_acquisition_does_not_conflict(tmp_path):
    path = tmp_path / "coordination.db"
    a_store, b_store = ProjectCoordinationStore(path), ProjectCoordinationStore(path)
    a, b = auth(), auth("bob")
    a_store.join(auth=a, kind="human")
    b_store.join(auth=b, kind="agent")
    acquired = []
    errors = []

    def acquire(store, resource, principal):
        try:
            acquired.append(store.acquire(resource, auth=principal))
        except Exception as exc:  # noqa: BLE001 - record all concurrent failures
            errors.append(exc)

    threads = [threading.Thread(target=acquire, args=(a_store, "r1", a)),
               threading.Thread(target=acquire, args=(b_store, "r2", b))]
    for worker in threads:
        worker.start()
    for worker in threads:
        worker.join()
    assert not errors
    assert {item["resource_id"] for item in acquired} == {"r1", "r2"}
    a_store.close()
    b_store.close()


def test_live_policy_revocation_blocks_lease_renewal_and_event_reads(tmp_path):
    revoked = set()
    store = ProjectCoordinationStore(tmp_path / "coordination.db")
    alice = AuthorizationContext("alice", "p1", frozenset({"collaboration:join", "collaboration:read",
                          "collaboration:lock"}), live_scope_check=lambda actor, project, scope:
                          (actor, project, scope) not in revoked)
    store.join(auth=alice, kind="human")
    lease = store.acquire("drawing", auth=alice)
    revoked.add(("alice", "p1", "collaboration:lock"))
    with pytest.raises(WorkflowAuthorizationError, match="current project policy"):
        store.renew("drawing", lease["lease_id"], lease["fencing_token"], auth=alice)
    revoked.add(("alice", "p1", "collaboration:read"))
    with pytest.raises(WorkflowAuthorizationError):
        store.events(auth=alice)
    store.close()


def test_collaborative_agent_uses_injected_host_store_without_owning_it(tmp_path):
    store = ProjectCoordinationStore(tmp_path / "coordination.db")
    agent = CollaborativeAgent(
        shared_memory=None, agent_factory=None, project_store=store,
        config={"use_collaboration_manager": False, "shared_memory_enabled": False,
                "publish_status": False},
    )
    alice = auth()
    assert agent.join_project(auth=alice, kind="human")["project_id"] == "p1"
    lease = agent.acquire_resource_lease("resource", auth=alice)
    assert agent.project_snapshot(auth=alice)["leases"][0]["resource_id"] == "resource"
    assert agent.project_events(auth=alice)[-1]["event_type"] == "lease.acquired"
    agent.release_resource_lease("resource", lease["lease_id"], lease["fencing_token"], auth=alice)
    agent.shutdown()
    # Host still controls the store and may reuse it across agent instances.
    assert store.events(auth=alice)[-1]["event_type"] == "lease.released"
    store.close()


# General-purpose inter-agent exchange tests live alongside the existing
# project-coordination tests: no separate communication test subsystem.
_EXCHANGE_SCOPES = frozenset({"collaboration:join", "collaboration:read",
                             "collaboration:message", "collaboration:handoff"})


def exchange_auth(actor, project="p1", scopes=_EXCHANGE_SCOPES):
    return AuthorizationContext(actor, project, scopes)


def join_agents(store, *actors):
    for actor in actors:
        store.join(auth=exchange_auth(actor), kind="agent")


def test_inter_agent_messages_recover_after_restart_and_ack_only_once(tmp_path):
    path = tmp_path / "exchange.db"
    first = ProjectCoordinationStore(path)
    join_agents(first, "a", "b", "c")
    message = first.send_message("b", "REQUEST", {"action": "analyze"}, auth=exchange_auth("a"),
                                 task_id="task-1", correlation_id="conversation-1",
                                 context_refs=["artifact:revision-1"], idempotency_key="request-1")
    assert message["sender_id"] == "a"
    assert message["schema_version"] == "1.0"
    assert message["context_refs"] == ["artifact:revision-1"]
    assert first.inbox(auth=exchange_auth("c")) == []
    assert first.send_message("b", "REQUEST", {"action": "analyze"}, auth=exchange_auth("a"),
                              task_id="task-1", correlation_id="conversation-1",
                              context_refs=["artifact:revision-1"],
                              idempotency_key="request-1")["message_id"] == message["message_id"]
    with pytest.raises(ProjectCoordinationError, match="idempotency key"):
        first.send_message("b", "REQUEST", {"action": "different"}, auth=exchange_auth("a"),
                           idempotency_key="request-1")
    with pytest.raises(ProjectCoordinationError, match="recipient inbox"):
        first.acknowledge_message(message["message_id"], auth=exchange_auth("c"))
    first.close()

    reopened = ProjectCoordinationStore(path)
    assert reopened.inbox(auth=exchange_auth("b"))[0]["message_id"] == message["message_id"]
    ack = reopened.acknowledge_message(message["message_id"], auth=exchange_auth("b"))
    assert ack["acked_at"] is not None
    assert reopened.acknowledge_message(message["message_id"], auth=exchange_auth("b"))["acked_at"] == ack["acked_at"]
    assert reopened.inbox(auth=exchange_auth("b")) == []
    assert [event["event_type"] for event in reopened.events(auth=exchange_auth("b"))].count(
        "message.acknowledged") == 1
    reopened.close()


def test_inter_agent_handoff_ownership_and_terminal_guards(tmp_path):
    store = ProjectCoordinationStore(tmp_path / "handoffs.db")
    join_agents(store, "a", "b", "c")
    pending = store.prepare_handoff("b", "task-1", "verify", "Review result", auth=exchange_auth("a"),
                                    details={"completed_steps": ["analyze"], "remaining_steps": ["verify"]},
                                    context_refs=["artifact:revision-7"], idempotency_key="handoff-1")
    handoff_id = pending["handoff_id"]
    assert pending["status"] == "pending" and pending["owner_id"] == "a"
    assert store.handoff_inbox(auth=exchange_auth("b"))[0]["handoff_id"] == handoff_id
    assert store.handoff_inbox(auth=exchange_auth("c")) == []
    with pytest.raises(ProjectCoordinationError, match="active exclusive"):
        store.prepare_handoff("c", "task-1", "verify", "Duplicate", auth=exchange_auth("a"))
    with pytest.raises(ProjectCoordinationError, match="not found"):
        store.get_handoff(handoff_id, auth=exchange_auth("c"))
    with pytest.raises(ProjectCoordinationError, match="cannot perform"):
        store.transition_handoff(handoff_id, "accept", auth=exchange_auth("a"))
    accepted = store.transition_handoff(handoff_id, "accept", auth=exchange_auth("b"))
    assert accepted["status"] == "accepted" and accepted["owner_id"] == "b"
    assert store.handoff_inbox(auth=exchange_auth("b"), status="accepted")[0]["handoff_id"] == handoff_id
    with pytest.raises(ProjectCoordinationError, match="invalid handoff transition"):
        store.transition_handoff(handoff_id, "cancel", auth=exchange_auth("a"))
    completed = store.transition_handoff(handoff_id, "complete", auth=exchange_auth("b"),
                                         result_refs={"verification": "result:42"})
    assert completed["status"] == "completed"
    assert completed["result_refs"] == {"verification": "result:42"}
    with pytest.raises(ProjectCoordinationError, match="invalid handoff transition"):
        store.transition_handoff(handoff_id, "complete", auth=exchange_auth("b"))
    assert [event["event_type"] for event in store.events(auth=exchange_auth("a"))][-3:] == [
        "handoff.prepared", "handoff.accepted", "handoff.completed"]
    store.close()


def test_handoff_expiration_restart_and_replacement(tmp_path):
    now = [1000.0]
    path = tmp_path / "handoffs.db"
    first = ProjectCoordinationStore(path, clock=lambda: now[0])
    join_agents(first, "a", "b")
    old_id = first.prepare_handoff("b", "task", "compute", "Continue", auth=exchange_auth("a"), ttl=5)["handoff_id"]
    first.close()
    now[0] = 1006.0
    recovered = ProjectCoordinationStore(path, clock=lambda: now[0])
    assert recovered.get_handoff(old_id, auth=exchange_auth("b"))["status"] == "expired"
    assert recovered.handoff_inbox(auth=exchange_auth("b"), status="expired")[0]["handoff_id"] == old_id
    with pytest.raises(ProjectCoordinationError, match="invalid handoff transition"):
        recovered.transition_handoff(old_id, "accept", auth=exchange_auth("b"))
    new_id = recovered.prepare_handoff("b", "task", "compute", "Continue", auth=exchange_auth("a"))["handoff_id"]
    assert new_id != old_id
    recovered.transition_handoff(new_id, "reject", auth=exchange_auth("b"), reason="unavailable")
    assert recovered.get_handoff(new_id, auth=exchange_auth("a"))["status"] == "rejected"
    recovered.close()


def test_inter_agent_scope_isolation_validation_and_message_expiry(tmp_path):
    now = [100.0]
    store = ProjectCoordinationStore(tmp_path / "messages.db", clock=lambda: now[0])
    join_agents(store, "a", "b")
    store.join(auth=exchange_auth("other", "p2"), kind="agent")
    with pytest.raises(WorkflowAuthorizationError):
        store.send_message("b", "REQUEST", {}, auth=exchange_auth("a", scopes=frozenset()))
    with pytest.raises(ProjectCoordinationError, match="active project participant"):
        store.send_message("missing", "REQUEST", {}, auth=exchange_auth("a"))
    with pytest.raises(ProjectCoordinationError, match="JSON serializable"):
        store.send_message("b", "REQUEST", {"nan": float("nan")}, auth=exchange_auth("a"))
    with pytest.raises(ProjectCoordinationError, match="JSON serializable"):
        store.send_message("b", "REQUEST", {"object": object()}, auth=exchange_auth("a"))
    message_id = store.send_message("b", "REQUEST", {"ok": True}, auth=exchange_auth("a"), ttl=2)["message_id"]
    assert store.inbox(auth=exchange_auth("b"))[0]["message_id"] == message_id
    now[0] = 103.0
    assert store.inbox(auth=exchange_auth("b")) == []
    with pytest.raises(ProjectCoordinationError, match="expired"):
        store.acknowledge_message(message_id, auth=exchange_auth("b"))
    with pytest.raises(ProjectCoordinationError, match="recipient"):
        store.send_message("b", "REQUEST", {}, auth=exchange_auth("other", "p2"))
    store.close()


def test_parallel_handoff_acceptance_has_single_winner(tmp_path):
    path = tmp_path / "handoffs.db"
    first, second = ProjectCoordinationStore(path), ProjectCoordinationStore(path)
    join_agents(first, "a", "b")
    handoff_id = first.prepare_handoff("b", "task", "compute", "Compute", auth=exchange_auth("a"))["handoff_id"]
    outcomes = []

    def accept(store):
        try:
            outcomes.append(store.transition_handoff(handoff_id, "accept", auth=exchange_auth("b"))["status"])
        except ProjectCoordinationError:
            outcomes.append("rejected")

    workers = [threading.Thread(target=accept, args=(store,)) for store in (first, second)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert sorted(outcomes) == ["accepted", "rejected"]
    first.close()
    second.close()


class _MessageProbe:
    def __init__(self, *, fail=False):
        self.hints = []
        self.fail = fail

    def publish(self, channel, value):
        if self.fail:
            raise RuntimeError("transient delivery unavailable")
        self.hints.append((channel, value))


def test_collaborative_agent_exposes_exchange_without_new_facade(tmp_path):
    store = ProjectCoordinationStore(tmp_path / "integration.db")
    join_agents(store, "a", "b")
    agent = CollaborativeAgent(
        shared_memory=None, agent_factory=None, project_store=store,
        config={"use_collaboration_manager": False, "shared_memory_enabled": True,
                "publish_status": False},
    )
    probe = _MessageProbe()
    agent.shared_memory = probe
    message = agent.send_agent_message("b", "REQUEST", {"task": "generic"}, auth=exchange_auth("a"))
    assert agent.agent_message_inbox(auth=exchange_auth("b"))[0]["message_id"] == message["message_id"]
    assert probe.hints == [("inter_agent:p1:b", {"kind": "message", "record_id": message["message_id"]})]
    agent.acknowledge_agent_message(message["message_id"], auth=exchange_auth("b"))

    # Transient SharedMemory notification failure must not roll back committed work.
    agent.shared_memory = _MessageProbe(fail=True)
    second = agent.send_agent_message("b", "RESULT", {"ok": True}, auth=exchange_auth("a"))
    assert agent.agent_message_inbox(auth=exchange_auth("b"))[0]["message_id"] == second["message_id"]
    handoff = agent.prepare_agent_handoff("b", "t1", "verify", "Check output", auth=exchange_auth("a"))
    assert agent.agent_handoff_inbox(auth=exchange_auth("b"))[0]["handoff_id"] == handoff["handoff_id"]
    assert agent.transition_agent_handoff(handoff["handoff_id"], "accept", auth=exchange_auth("b"))["status"] == "accepted"
    assert agent.transition_agent_handoff(handoff["handoff_id"], "complete", auth=exchange_auth("b"))["status"] == "completed"
    agent.shutdown()
    assert store.acknowledge_message(second["message_id"], auth=exchange_auth("b"))["acked_at"] is not None
    store.close()
