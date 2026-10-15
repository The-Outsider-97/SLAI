"""Apply ADR additions to a checked-out SLAI-v2.3 repository.

Run from the repository root AFTER inspecting `git diff --stat` and branch.
Creates .adr-backup copies of the two files it changes. Refuses ambiguous edits.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

MANAGER = Path('src/agents/collaborative/collaboration_manager.py')
CONFIG = Path('src/agents/collaborative/configs/collaborative_config.yaml')

ORIGINAL = '''        self.strategy = strategy or build_router_strategy(strategy_name, config=self.routing_config)
        self.router = router or self._build_task_router()'''
REPLACEMENT = '''        self.strategy = strategy or build_router_strategy(strategy_name, config=self.routing_config)
        # ADR augments ranking only; registry, contracts, policy and execution remain unchanged.
        adr_config = get_config_section("delegation_ranker") or {}
        if adr_config.get("enabled", False) and router is None:
            from .delegation_ranker import DelegationRankerStrategy
            self.strategy = DelegationRankerStrategy(self.strategy, adr_config)
        self.router = router or self._build_task_router()
        if adr_config.get("enabled", False) and router is None:
            self.router.register_hook(self.strategy.on_route)'''

CONFIG_BLOCK = '''
# ADR rollout: activated collection in shadow mode requires enabled: true.
# Active mode must not be used without external live evaluation and approval.
delegation_ranker:
  enabled: false
  mode: "shadow"                 # disabled | shadow | advisory | active
  model_dir: "src/agents/collaborative/models/delegation_ranker_v1"
  event_db: "data/training/collaborative/delegation_ranker/raw/events.sqlite3"
  success_value: 1.0             # relative value of a completed task
  latency_penalty_per_ms: 0.0    # zero until latency labels support this head
  min_confidence: 0.0            # 0 permits all scores; 1 blocks uncertain results
  advisory_mix: 0.25
'''


def patch(repo: Path) -> None:
    manager = repo/MANAGER; config=repo/CONFIG
    if not manager.is_file() or not config.is_file():
        raise FileNotFoundError('Expected SLAI v2.3 Collaborative files missing')
    raw_manager = manager.read_bytes().decode('utf-8')
    manager_newline = '\r\n' if '\r\n' in raw_manager else '\n'
    source = raw_manager.replace('\r\n','\n')
    if 'self.router.register_hook(self.strategy.on_route)' in source:
        raise RuntimeError('ADR manager hook already present; refusing double application')
    if source.count(ORIGINAL)!=1:
        raise RuntimeError('Unsupported CollaborationManager version or unexpected constructor layout')
    raw_config = config.read_bytes().decode('utf-8')
    config_newline = '\r\n' if '\r\n' in raw_config else '\n'
    cfg = raw_config.replace('\r\n','\n')
    if '\ndelegation_ranker:' in cfg:
        raise RuntimeError('ADR configuration already exists')
    for f in (manager,config):
        backup=f.with_name(f.name+'.adr-backup')
        if backup.exists():
            raise FileExistsError(f'Backup would be overwritten: {backup}')
        shutil.copy2(f,backup)
    manager.write_bytes(source.replace(ORIGINAL,REPLACEMENT).replace('\n',manager_newline).encode('utf-8'))
    config.write_bytes((cfg.rstrip()+ '\n'+CONFIG_BLOCK).replace('\n',config_newline).encode('utf-8'))
    print('Patched:',MANAGER,CONFIG)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--allow-non-v23',action='store_true',help='Use only on verified SLAI v2.3 checkout')
    args=parser.parse_args();root=args.repo.resolve()
    if not args.allow_non_v23:
        result=subprocess.run(['git','branch','--show-current'],cwd=root,capture_output=True,text=True,check=True)
        if result.stdout.strip()!='SLAI-v.2.3':
            raise SystemExit('Refusing to patch branch other than SLAI-v.2.3')
    patch(root)

if __name__=='__main__':main()
