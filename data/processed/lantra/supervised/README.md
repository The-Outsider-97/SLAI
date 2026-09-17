# LANTRA supervised seed corpus — v2

This replacement bundle fixes the cross-split duplicate-content defect in the original seed corpus.

Each of the seven tasks contains 120 examples:
- 96 train
- 12 validation
- 12 test

Total: 840 examples.

Integrity audit:
- exact duplicate supervised contents: 0
- conflicting explicit splits: 0
- unique IDs: 840
- unique normalized content records: 840

Replace the previous files under:

`data/processed/lantra/supervised/`

Recommended command:

```powershell
py -m train_lantra `
  --init-from src\agents\language\checkpoints\lantra\lantra_20260917T162402Z.pt `
  --data data\processed\lantra\supervised `
  --raw-pretrain-epochs 0 `
  --no-glove-bootstrap `
  --require-all-supervised-tasks
```
