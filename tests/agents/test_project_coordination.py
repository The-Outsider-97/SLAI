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
        except Exception as exc:
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
