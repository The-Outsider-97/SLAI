# Secure Folder security model (version 0.1)

## Trust boundary

The vault protects encrypted content at rest from someone who obtains only a copy of the vault storage without the passphrase. It also protects routes from unauthenticated use through the normal local application interface.

It **does not** promise security from a fully compromised operating system, malware with the user's privileges, privileged process-memory inspection, keyloggers, or a user choosing to export plaintext. The local browser interface is not an OS-level secure enclave or a sandboxed mobile app.

## Encryption and keys

1. First setup creates a fresh 256-bit random master key and a random 128-bit scrypt salt.
2. `scrypt` (N=32768, r=8, p=1) derives a 256-bit wrapping key from the user-entered passphrase; the master key is wrapped with AES-256-GCM and authenticated associated data.
3. `vault.json` stores only the version, KDF parameters identifier, salt and encrypted master key.
4. `index.enc` contains the encrypted, authenticated JSON manifest of all file names, parent relationships, notes, sizes, favorites and trash state.
5. Each file is encrypted into bounded chunks using fresh cryptographically random AES-GCM nonces and associated data binding each chunk to its random blob ID and index. On disk, blob IDs are random rather than derived from user filenames.
6. Integrity is checked on every chunk decryption, with a separate SHA-256 checksum of the reconstructed plaintext file checked by the current browser client on complete read.
7. Passphrase changes rewrap the master key and revoke all sessions. Passphrases and unwrapped keys are not persisted.

The selected scrypt KDF is a memory-hard alternative to Argon2id already supported by the project's `cryptography` dependency. It is *not* a hardware-backed anti-guessing control: a thief with an offline copy of the vault can attempt password guesses. Use a long, unpredictable passphrase.

## Sessions and network

- SLAI's shared loopback-only HTTP server independently checks a launch token cookie, Host/Origin headers, a marker header on mutations, CSP, no-store and file-path restrictions.
- Secure Folder adds independent random volatile unlock tokens in the `X-Vault-Session` header. Possession of the launch cookie alone cannot read vault contents.
- The client never saves an unlock token in a cookie, URL, browser local storage or session storage.
- The server expires unlock tokens after 5 minutes without a vault operation or 30 minutes in total; the client also locks on inactivity and backgrounding. A failed browser teardown may leave a volatile server token until timeout, but reopening still requires the passphrase.
- Password unlock failures are delayed with an in-process increasing backoff. Rate limits reset on service restart and cannot defeat offline password guessing.
- The application uses an OS-level advisory single-instance file lock to prevent two local servers writing the same vault index.
- The vault denies agent access by default by exposing no AI/tool routes.

## Known limitations and risks

- **Security review pending:** This is a development implementation, not an independently audited or certified vault.
- **Backup and recovery missing:** There is not yet a supported encrypted backup/restore workflow. Do not store irreplaceable data without another safe copy.
- **Export limitations:** Export requires passphrase confirmation in the UI, but an already-unlocked, authorized client can legitimately request plaintext chunks for preview. A browser cannot reliably prevent an authorized viewer from saving/copying bytes. Preview/export is buffered and limited to 96 MiB.
- **Mobile packaging:** No native Android/iOS release, device-keystore integration, biometric isolation or platform screenshot restrictions yet. The current UI can adapt to phone-sized screens only when the local server is available on that device.
- **No hardware-backed PIN:** Short PINs are explicitly disabled because software-only offline encryption would be weak against guessing.
- **Temporary plaintext:** Browser memory necessarily contains decrypted previews; OS swap, screenshots, browser extensions, crash dumps, and user downloads cannot be cryptographically controlled by a web UI. No HTML/SVG preview is permitted.
- **Deletion:** Permanent delete removes encrypted blobs logically. SSD and backup forensics can retain old ciphertext; physical secure erasure is not guaranteed.
- **Orphaned ciphertext:** An interrupted upload may leave encrypted chunks that are not referenced by the manifest. A future maintenance operation should safely clean stale blobs.
- **No native document viewer integration:** `.sldc`, `.sgeo`, `.sdrw`, PDF/audio/video preview and sandboxed handoff still need reviewed adapters.
- **No MFA or secure audit trail:** These are intentionally not represented by fake switches or misleading status indicators.
- **Service executable protection:** A malicious local user with privileged access could change application source code or inspect running memory; this prototype is not a defense against an adversarial host.

## Recommendations before production

1. Independent security review of key lifecycle, import/restore handling, concurrency and all HTTP entry points.
2. Versioned encrypted backup and tested atomic restore/migration, with corruption/recovery drills.
3. Stronger platform-specific rate limiting, OS-backed key handling, native mobile sandboxing and lifecycle enforcement.
4. Streaming preview/export with bounded memory and controlled viewer adapters.
5. Test on physical Windows, macOS, Linux, Android and iOS targets; verify background lock and sleep behavior.
6. Dependency vulnerability review and supply-chain hardening, including signing distributed binaries.
