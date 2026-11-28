# SLAI Communication

## Purpose and verified baseline

A compact, additive infrastructure subsystem for canonical structured inter-agent messages, native encrypted SLDC journaling, scoped human inspection, access auditing and archive key lifecycle. It is not an agent, message router, new file format or replacement for Governance, Privacy, Shared Memory, Secure Memory, AgentFactory, Orchestrator, Network or LSTP.

Repository baseline checked on 28 November 2026: all 18 visible branch heads were compared. `SLAI-v.2.5` at `3d63e7ca45923d37cd3232a59223ea4532e2fae7` (26 November 2026) was newer than `main` at `f4d8dff9bea64d719c43186dde8fed6fbef53991` (15 November). No GitHub releases were returned; tags could not be enumerated through the connector. Native `src/utils/sldc.py` blob `7fb8357ecbabf867713edf95e4ebbebe366ef75d` exposes package creation, X25519 and password recipients, Ed25519 signatures, encrypted reads/writes and inspection. `requirements.txt` already includes `cryptography`. The newer branch also includes `src/agents/governance/agent_gate.py`, `delegation_guard.py`, `effect_boundary.py`, and `utils/identity.py`. These are existing runtime authorities; this package does not claim integration with them.

## Layout

```text
src/communication/
  __init__.py                 # public sender-bound API
  __main__.py                 # trusted local operator CLI
  communication_types.py      # validated message schema
  communication_manager.py    # runtime policy, sanitization and delivery callbacks
  communication_security.py   # account scopes, credentials, epochs, state lock
  communication_storage.py    # signed SLDC journal and trusted read/audit facade
  communication_service.py    # host-provided Linux Unix socket, read-only protocol
  README.md                   # this document
```

No new runtime dependencies, custom cryptographic algorithms, parallel event bus or new persistence format. SLDC may use JSON internally in its native container; there are no standalone JSON communication files. The message schema `slai.communication.v1` is unchanged. The service state, account vault and access audit schemas are versioned independently as `v2`; the earlier package's shared-key vault architecture is intentionally incompatible.

## Trust model — essential before deployment

**Only a trusted service/operator OS identity may read or write the service directory or obtain the `CommunicationJournal`/`KeyStore` objects.** Human users authenticate with username/password via a trusted adapter that exposes only `view()`. Account vaults contain an authentication token, not archive decryption keys. The token is checked against the protected state; no plaintext password is stored. The local CLI is for trusted operators and testing: running it directly as an untrusted OS user with access to `service.env` would bypass the security boundary. Do not distribute that file, the state file, or private service objects to readers. This package does not include an authenticated network server or GUI. A production adapter must authenticate requests, rate-limit, protect transport, avoid password/output logging and bind the actual SLAI identity and Governance policy. In-process Python code is not an isolation boundary.

Admin methods are intentionally service-only, not granted to ordinary human reading accounts. The CLI records the effective POSIX UID or Windows domain-qualified account from the OS. The optional `operator_id` argument must equal that identity. Hosts that supplied an application-level name must omit it or supply the matching OS principal. A shared service identity cannot identify the individual administrator behind a request; step-up and approval integration remain outstanding. No role is inferred from usernames (e.g. `board` or `auditor`).

Use a directory outside the Git checkout with restrictive NTFS ACLs (Windows) or owner-only mode (POSIX). POSIX permissions are checked for directory and service credential file; Windows ACLs are **not verified by the code**. Set and test them under the actual OS identities before use. Protect backups, terminal output, crash dumps and process environment. The root fingerprint is public but must be pinned independently. Root/state keys reside in OS-protected `service.env`; they are not human passwords. They need a secure OS key store or equivalent hardened service deployment for production.

## Installation and initial configuration (PowerShell)

From a checkout of the target branch, use its existing Python environment with the repository cryptography dependency installed. Run the native suites before initializing service storage.

```powershell
Set-Location 'G:\GAIA\AI\SLAI'
python -m unittest discover -s tests -p 'test_communication*.py' -v
python -m unittest discover -s tests -p 'test_sldc.py' -v
```

For a **new** service directory (parent must exist), use the same OS identity for setup and local tests:

```powershell
cd G:\GAIA\AI\SLAI
$Service = Join-Path $env:LOCALAPPDATA 'SLAI-communication-service'
python -m src.communication --service-dir "$Service" init
```

The command returns the root signing fingerprint; verify and pin it independently, then set the public pin for subsequent commands:

```powershell
$env:SLAI_COMM_TRUSTED_ROOT = '<verified fingerprint>'
```

Service directory:

```text
service.env                  # private root signing seed and state encryption keypair
state.sldc                   # encrypted, root-signed account/epoch state
.lock                        # cooperating-process advisory lock
accounts/<user>-<generation>.sldc   # password-encrypted, root-signed authentication vault
journal/<epoch>/<message-id>.sldc    # archive-encrypted, epoch-signed immutable message
audit/<epoch>/<event-id>.sldc       # independently audit-encrypted, root-signed event
```

There is no permissive/default user or automatic full-access account. Store credentials outside source control. Backup the **entire** service directory coherently while stopped or under an exclusive lock, and retain the root pin separately. Restore tests are required before reliance on backups; rollback to a stale backup can revive revoked credentials and keys, so quarantine and reconcile against independent state/audit anchors before reopening.

## Scoped human access

Four dimensions are required and combined with logical AND. Within a dimension values are ORed. `['*']` is an explicit unrestricted wildcard; `[]` denies the dimension. The agent dimension matches either sender or recipient. Classifications are exact values (`internal`, `confidential`, `restricted`), not implicit clearance hierarchies. `kinds` uses the existing eleven communication kinds. Scope changes take effect on the next authenticated read; no cached human sessions are provided. Query filters further narrow the scope and out-of-scope explicit filters are denied. A trusted adapter may pass a Governance read-policy callback; it receives only already-scoped records and can deny but not grant beyond scope. A callback is not a substitute for OS isolation or end-to-end Governance integration.

```powershell
# Explicitly full-reading Board account; not a service administrator
python -m src.communication --service-dir "$Service" add-user --user board --agents '*' --sessions '*' --kinds '*' --classifications '*'
# Narrow Provenance account
python -m src.communication --service-dir "$Service" add-user --user provenance_auditor --agents provenance_agent --sessions '*' --kinds provenance --classifications internal
# Replace permissions (all four dimensions required)
python -m src.communication --service-dir "$Service" set-scope --user provenance_auditor --agents provenance_agent --sessions project_1 --kinds provenance --classifications internal
# View with own credentials, subject to scope
python -m src.communication --service-dir "$Service" view --user provenance_auditor --agent provenance_agent --limit 50
```

The operator CLI accepts exact filters `--sender`, `--recipient`, `--task`, `--correlation`, `--classification` and `--record-id`. Filters intersect with current scope and the existing agent, kind and session filters. Authorization runs again after journal commit immediately before delivery. A policy change at that point raises `DeliveryError` with the archived message ID and suppresses dispatch. Host policies must support repeated checks.

On POSIX, Communication propagates directory sync errors for journal, audit, state and credential writes. A sync failure after publication leaves the outcome uncertain. Reconcile the existing record or administrative intent before retrying. Windows power-loss durability and filesystem behavior require deployment testing.

A password is requested interactively. `view` returns up to 1000 records per request (default 100). The `next_cursor` is a position, not a credential, and is rechecked against current permissions. Pass it to `--after` for the next page. Ordering is lexicographic by epoch and message UUID, **not chronological**. Pages are not a stable snapshot; concurrent additions before the cursor may be missed. Human reads omit revoked epochs, do not expose unauthorized counts, and never return audit/state/private credentials. The caller must not provide its own policy callback or be permitted to invoke arbitrary service methods.

## Key lifecycle and revocation

```powershell
python -m src.communication --service-dir "$Service" status
python -m src.communication --service-dir "$Service" rotate-keys
python -m src.communication --service-dir "$Service" rotate-password --user provenance_auditor
python -m src.communication --service-dir "$Service" revoke-user --user provenance_auditor
# Deliberate reactivation only; re-check scope before reactivating.
python -m src.communication --service-dir "$Service" rotate-password --user provenance_auditor --reactivate
# If a journal epoch is suspected compromised, use its ID from status:
python -m src.communication --service-dir "$Service" revoke-epoch --epoch '<epoch-id>'
```

- **Active:** one epoch encrypts and signs new communications; independent X25519 keys encrypt that epoch's audit records.
- **Retired:** normal rotation creates a new epoch, retires the prior one, removes the prior signing seed from current state, and retains decryption keys and trusted signer fingerprint for historical records. No bulk rewrite or password prompts are required. Stale service instances reload current state under the lock before writing or reading.
- **Revoked:** normal human reads skip the epoch completely. Revoking the current epoch atomically creates a replacement. Records and decryption keys remain available to trusted operators for preservation/forensics; this is quarantine, not deletion. Audit records remain readable by the trusted operator using separate audit encryption keys and root signatures.
- **Account revocation/password rotation:** new account generations replace old tokens; the current signed state selects the valid generation. Revocation immediately denies subsequent requests in cooperating service processes. Old password vaults remain encrypted on disk but are no longer accepted by current state. Reactivation explicitly creates a new generation and preserves assigned scope. Previously disclosed plaintext cannot be revoked.

Root signing/state encryption keys are **not** rotated by `rotate-keys`; these anchor the control plane. If root or state key compromise is suspected, stop the service, isolate it, preserve evidence and conduct a separately approved trust-root migration/re-encryption with independent verification. Automated root migration, HSM integration, cryptographic erasure and multi-party approval are not included. Retired private encryption keys are retained for historical access; their compromise exposes corresponding ciphertext. Removing a seed from active state does not erase backups or memory. A compromised old signing seed can forge old records until the epoch is revoked; signatures alone cannot establish trustworthy historical time or completeness.

## Access audit

The service writes immutable signed, encrypted SLDC events for authentication success/failure, scope/policy decisions, query/integrity errors, exact returned record IDs, operator status/audit reads, and administrative change intent/completion. Audits omit passwords, raw message payloads and raw query strings. They contain account/record identifiers and therefore remain sensitive. The number of filtered records is recorded only in the operator audit. Audit encryption is independent of communication encryption. Audit write failure prevents a human read from returning any records. Failed attempts before state can be loaded or before a request reaches the service cannot be recorded by this journal; use host monitoring as well.

```powershell
python -m src.communication --service-dir "$Service" audit --limit 100
```

An administrative change records intent, atomically writes state, then writes completion. A state-write error attempts to record an uncertainty event against the actually readable state. A crash or failure between these operations can leave an intent without completion; inspect `status` and reconcile before retrying. Audit records are individually tamper-evident, not a cryptographically complete sequence: deletion, rollback or filesystem bypass are not independently detected. Add external signed checkpoints/anchors and retention/hold policy before regulated production deployment. Audit reading is itself audited; paginating a live audit can encounter new audit records. No audit export endpoint is included.

## Runtime integration

The previous `AgentCommunication` send API is preserved. The journal constructor now takes service directory and independent root fingerprint rather than a fixed public encryption key and signer. It reloads active keys for every write, avoiding stale-key writes after rotation. All authorization, classification, privacy and actual delivery remain with existing SLAI authorities. Example below uses placeholders for runtime adapters, not invented existing methods:

```python
from pathlib import Path
from src.communication import AgentCommunication
from src.communication.communication_storage import CommunicationJournal

journal = CommunicationJournal(Path(r'C:\private\SLAI-communication-service'), trusted_root_pin)
client = AgentCommunication(
    sender='provenance_agent',
    journal=journal,
    authorize=verified_governance_adapter,  # exact sanitized record -> bool
    sanitize=verified_privacy_adapter,      # record -> validated record
    deliver=verified_runtime_router,      # recipient, record -> None
)
message_id = client.send(
    'orchestrator_agent', 'provenance', {'artifact_id': 'example'},
    session_id='project_1', classification='internal',
)
```

`AgentCommunication` validates, sanitizes, authorizes, durably records, then invokes delivery. A `DeliveryError` means recorded but delivery failed or is uncertain; receiver acknowledgments and idempotent retry are host responsibilities. Use references for large/sensitive assets. Do not send raw credentials, hidden reasoning, or decrypted restricted payloads into uncontrolled SharedMemory broadcasts. Native Governance gate and identity APIs must be integrated and tested against the latest branch, not blindly mapped to these illustrative callbacks. Avoid invoking reentrant storage calls from policy callbacks while the journal lock is held.

Linux service hosts can instantiate `CommunicationReaderService(journal, uid_to_account, authorize_read)` and pass an accepted stream to `serve_connection(connection)`. The host owns admission limits, the protected socket directory, worker pool, socket ACLs, authoritative policy and service identity. No listener starts on import. Each request is one UTF-8 JSON object ending in a newline, at most 8192 bytes, with `password` and optional query fields from `view`. The page size is at most 25. The service resolves the account from kernel peer credentials and a one-to-one UID mapping. It ignores environment usernames and rejects client-supplied account names, callbacks, paths and administrative actions. Every accepted query reauthenticates under the state lock. Scope and the host's mandatory per-record policy apply before disclosure. Successful replies carry `ok`, `records` and `next_cursor`. All rejected requests return only `{"ok":false}`. The service never logs the password. Hosts must restrict access to the socket, disable credential capture and avoid sharing reader UIDs.

Failed passwords for mapped local readers trigger a persisted delay of 1–64 seconds. Attempts during the delay cannot extend it. State is shared by cooperating processes with the same journal and clock. A successful authentication clears the throttle. Direct operator calls are outside this broker. Deployment-wide abuse detection, failover across independent hosts, transport on Windows and authenticated export are still open.

## Upgrade from the previous package and legacy JSON

**Do not run `init` against an old vault or delete legacy keys.** Prior account vaults shared an archive decryption key, so rotating their passwords cannot revoke keys already obtained. The new state/account/epoch schemas are intentionally incompatible. `-Upgrade` replaces only source, not old journals or vaults. Establish a fresh private service directory; preserve the original vault/journal and independent old signer pin. Revoke old filesystem access; assess any previously copied keys as exposed. An approved migration must verify old encrypted signatures and exact message schemas, import under the new key epoch with stable message IDs and record counts, verify historical access permissions and integrity, and securely retire or quarantine old material only after acceptance. No automatic migration is provided because old key distribution cannot be inferred. Legacy JSON-backed domain state (including Provenance) must be migrated through its authoritative persistence adapter, not dumped into this communication journal. Preserve domain queries, graph edges, checkpoints and rollback.

## Verification and release gates

Run in the actual latest SLAI checkout:

```powershell
python -m unittest discover -s tests -p test_communication_unit.py -v
python -m unittest discover -s tests -p test_communication_native.py -v
```

Isolated tests exercise sender binding, policy denial, sanitization and storage/delivery errors. Native tests exercise scoped reads, password-protected accounts, no archive key in vault, rotation, stale instance behavior, revocation, restart, duplicate IDs, audit content and failure-before-disclosure, tamper rejection, independent audit encryption, named operator attribution, cooperating-thread rotation and pagination. The packaging environment could read the repository via GitHub but could not download native SLDC into its runtime; native crypto tests remain unexecuted against that code. See package README for actual verification results.

Remaining gates: native SLDC and Windows lock/ACL verification, actual Governance and identity integration, all-agent/application coverage, service deployment and rate limiting, rollback/deletion detection and external trust anchors, named admin audit, retention and deletion workflows, migration, crash/power-loss and concurrent stress tests, performance benchmarks (the lock serializes reads and writes; state is reloaded and signed SLDC files are individually written), and end-to-end compliance assessment. The small concurrent-thread test does not replace multi-process or Windows stress verification. No claim of production readiness or regulatory certification is made.

## Incremental filesystem hardening and verification

The service now checks ownership, owner-only permissions, regular-file type and single-link status on opened POSIX service files, including the lock, account vaults, state and journals. Final-component symlinks are rejected at open and FIFOs do not block the reader. Reads are bounded by the opened file size and reject a size change. On Windows, final-component reparse checks are best effort and do not verify NTFS ACLs or prevent all path races. Parent-directory path traversal, trusted service deployment, credential storage, independent anchors and full OS-identity tests remain release blockers. Authentication errors suppress the underlying exception chain; cross-host throttling and timing protection remain pending.

The repository now includes isolated facade regressions and native SLDC tests at the two documented test paths. Native execution and full repository regression verification remain mandatory; adding tests is not evidence that they passed. No production data or keys were migrated or rotated by this change.


## Managed directory hardening and process-level regression

The service rejects a symlink/reparse point at the service root or any managed accounts, journal, audit or epoch directory. On POSIX these directories must also be owned by the effective service user and have no group/other permissions. Managed root and area checks run before and after acquiring the service lock; epoch checks run before journal/audit traversal and writes. These checks do not defend against an attacker able to replace ancestor directories or race the trusted service identity; deploy with a protected parent and an isolated service user. Missing lazy epoch directories still cannot be distinguished from deletion without an independent anchor. Windows reparse checks are not NTFS ACL verification.

The native suite additionally covers directory substitution and a spawn-based two-process duplicate-message race against the actual SLDC engine. These tests are release evidence only once executed successfully. Full native, multi-process stress, Windows ACL, external anchor, governance, service and other architectural gates remain open.


## Release evidence for the update based on 4a3cd33

| Requirement | Evidence and remaining gate |
| --- | --- |
| P0.1 native reliability | Native archive rotation, revocation, duplicate-process race, failed audit and POSIX directory-sync tests. Physical power-cut recovery and Windows results outstanding. |
| P0.2 runtime policy | Exact record checked before append and immediately before dispatch. Host identity, trusted classification, Privacy and effects integration across runtime paths outstanding. |
| P0.3 human access | Linux socket-pair integration covers successful read, peer mapping, policy denial, malformed operation, backoff and revocation. Host deployment, export and Windows broker outstanding. |
| P0.4 OS isolation | Private POSIX inode and directory tests pass. Separate-user deployment, NTFS DACL verification, backup and secret-store provisioning outstanding. |
| P0.5 migration | Current ProvenanceStore uses its JSON adapter. No approved legacy cutover, signed migration manifest or rollback round trip. |
| P1.1 root/state keys | Operational epochs tested. Independent root succession and recoverable state-key migration outstanding. |
| P1.2 audit continuity | Signed events and durable append tested. No independent witness for deletion, truncation or state rollback. |
| P1.3 administrators | Local audit binds to the effective OS principal. Named human delegation behind shared service accounts, step-up and dual control outstanding. |
| P1.4 authentication | Durable bounded backoff per kernel-bound local reader. Cross-host protection, audit growth under abuse and operational recovery outstanding. |
| P1.5 retention | Deployment policy, holds, erasure across backups and restore reconciliation outstanding. |
| P1.6 agents and apps | BaseAgent.broadcast writes directly to Shared Memory. AgentFactory, Orchestrator, dynamic subagents and CADCon lack verified end-to-end Communication integration. |
| P2.1 search | Exact filters and four-dimensional scope tested. Chronological snapshot pagination, date ranges and rebuildable protected index outstanding. |
| P2.2 performance | See benchmarks/communication for both raw samples and reproduction script. Single Linux worker, 25 records, 25 reads of 10. Median append 2.38 to 2.24 ms and read 332.94 to 330.23 ms. No performance conclusion from one paired run. Sustained concurrency and resource profiling outstanding. |
| P2.3 graphical inspection | No integrated graphical application verified. |

Local verification used Python 3.12.14 and cryptography 50.0.0. All 17 Communication and 148 SLDC engine tests passed. No runtime dependency was added. To repeat the measurement from the repository root, set PYTHONPATH to the checkout under test and execute `python benchmarks/communication/bench_journal.py --records 25`. Baseline branch discovery found 18 heads and no advertised tags. This update targets the explicitly requested branch.
