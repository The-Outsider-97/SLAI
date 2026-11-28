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
  README.md                   # this document
```

No new runtime dependencies, custom cryptographic algorithms, parallel event bus or new persistence format. SLDC may use JSON internally in its native container; there are no standalone JSON communication files. The message schema `slai.communication.v1` is unchanged. The service state, account vault and access audit schemas are versioned independently as `v2`; the earlier package's shared-key vault architecture is intentionally incompatible.

## Trust model — essential before deployment

**Only a trusted service/operator OS identity may read or write the service directory or obtain the `CommunicationJournal`/`KeyStore` objects.** Human users authenticate with username/password via a trusted adapter that exposes only `view()`. Account vaults contain an authentication token, not archive decryption keys. The token is checked against the protected state; no plaintext password is stored. The local CLI is for trusted operators and testing: running it directly as an untrusted OS user with access to `service.env` would bypass the security boundary. Do not distribute that file, the state file, or private service objects to readers. This package does not include an authenticated network server or GUI. A production adapter must authenticate requests, rate-limit, protect transport, avoid password/output logging and bind the actual SLAI identity and Governance policy. In-process Python code is not an isolation boundary.

Admin methods are intentionally service-only, not granted to ordinary human reading accounts. The current CLI records the actor as `local-operator`. A trusted composition root can bind a verified named operator using `CommunicationJournal(..., operator_id=verified_identity)`; this parameter is not an authentication mechanism and must never be supplied by an untrusted client. Deployment must bind named administrator identities to the audited management interface or use OS accountability. No role is inferred from usernames (e.g. `board` or `auditor`).

Use a directory outside the Git checkout with restrictive NTFS ACLs (Windows) or owner-only mode (POSIX). POSIX permissions are checked for directory and service credential file; Windows ACLs are **not verified by the code**. Set and test them under the actual OS identities before use. Protect backups, terminal output, crash dumps and process environment. The root fingerprint is public but must be pinned independently. Root/state keys reside in OS-protected `service.env`; they are not human passwords. They need a secure OS key store or equivalent hardened service deployment for production.

## Installation and initial configuration (PowerShell)

Extract the full package. Inspect the current repository branch/commit and compare with the baseline above. Activate the existing virtual environment, then:

```powershell
& 'C:\Downloads\SLAI_communication_enhanced\install_communication.ps1' -SLAIPath 'G:\GAIA\AI\SLAI'
# If you already installed the previous source package, explicitly opt in:
& 'C:\Downloads\SLAI_communication_enhanced\install_communication.ps1' -SLAIPath 'G:\GAIA\AI\SLAI' -Upgrade
```

The installer backs up the named source files/tests on `-Upgrade`, copies only packaged files, and runs isolated and native tests. It does not change repository branches, commit/push, migrate or delete runtime data. If a test fails, stop and inspect the source backup; it does not automatically roll back. PowerShell itself was not executable in the packaging environment.

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

Human service adapter (illustrative, trusted process only): `journal.view(username, password, authorize_read=verified_governance_read_adapter, limit=100)`. The scope is always checked. The broker must never accept client-supplied policy, keys, paths, or arbitrary method names. No network service is provided; implement or connect to a real SLAI authenticated access surface before allowing untrusted clients.

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

The service now checks ownership, owner-only permissions, regular-file type and single-link status on opened POSIX service files, including the lock, account vaults, state and journals. Final-component symlinks are rejected at open and FIFOs do not block the reader. Reads are bounded by the opened file size and reject a size change. On Windows, final-component reparse checks are best effort and do not verify NTFS ACLs or prevent all path races. Parent-directory path traversal, trusted service deployment, credential storage, independent anchors and full OS-identity tests remain release blockers. Authentication errors suppress the underlying exception chain; timing protection and distributed rate limiting remain pending.

The repository now includes isolated facade regressions and native SLDC tests at the two documented test paths. Native execution and full repository regression verification remain mandatory; adding tests is not evidence that they passed. No production data or keys were migrated or rotated by this change.
