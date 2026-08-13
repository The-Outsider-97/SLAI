# Branch protection for `SLAI-v.2.2`

Apply protection only after `.github/workflows/ci.yml` has been pushed and both
CI jobs have completed successfully. GitHub only allows a status check to be
selected as required after that check has run successfully in the repository.

## Recommended rule

Create one branch rule or ruleset targeting exactly `SLAI-v.2.2` with:

- Require a pull request before merging.
- Required approvals: `0` for a single-maintainer repository, or `1` when an
  independent reviewer is consistently available.
- Dismiss stale approvals when new commits are pushed: enabled when approvals
  are required.
- Require conversation resolution before merging.
- Require status checks to pass before merging.
- Require branches to be up to date before merging (strict checks).
- Required checks:
  - `Foundation CI`
  - `Python 3.12 compatibility`
- Require linear history only if squash merge or rebase merge is enabled.
- Enforce the rule for administrators. If a bypass is retained for emergency
  recovery, restrict it to repository administrators and document its use.
- Block force pushes.
- Block branch deletion.

Do not enable signed-commit enforcement, deployment gates, merge queues, code
owner approval, or push restrictions until the repository actually uses those
workflows. Enabling inactive controls would create friction without adding a
working safety boundary.

## Activation order

1. Commit and push the CI and dependency-management files.
2. Let both checks complete successfully on `SLAI-v.2.2`.
3. Open **Settings â†’ Rules â†’ Rulesets** (preferred) or **Settings â†’ Branches**.
4. Create the rule above and select the two checks by their exact names.
5. Open a small test pull request and verify that merging is blocked until both
   checks pass.
