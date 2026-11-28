"""Local trusted-operator CLI. For remote/untrusted users use a separate authenticated service."""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

from .communication_storage import CommunicationJournal


def _scope(parser):
    parser.add_argument('--agents', nargs='+', required=True)
    parser.add_argument('--sessions', nargs='+', required=True)
    parser.add_argument('--kinds', nargs='+', required=True)
    parser.add_argument('--classifications', nargs='+', required=True)


def _passphrase(user):
    value = getpass.getpass(f'New passphrase for {user}: ')
    if value != getpass.getpass('Confirm passphrase: '):
        raise ValueError('Passphrases do not match')
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service-dir', type=Path, required=True)
    parser.add_argument('--trusted-root', default=os.getenv('SLAI_COMM_TRUSTED_ROOT'))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    add = sub.add_parser('add-user'); add.add_argument('--user', required=True); _scope(add)
    scope = sub.add_parser('set-scope'); scope.add_argument('--user', required=True); _scope(scope)
    password = sub.add_parser('rotate-password'); password.add_argument('--user', required=True); password.add_argument('--reactivate', action='store_true')
    revoke = sub.add_parser('revoke-user'); revoke.add_argument('--user', required=True)
    sub.add_parser('rotate-keys')
    epoch = sub.add_parser('revoke-epoch'); epoch.add_argument('--epoch', required=True)
    sub.add_parser('status')
    view = sub.add_parser('view'); view.add_argument('--user', required=True)
    for flag in ('agent', 'session', 'kind', 'after'): view.add_argument('--'+flag)
    view.add_argument('--limit', type=int, default=100)
    audit = sub.add_parser('audit'); audit.add_argument('--limit', type=int, default=100); audit.add_argument('--after')
    args = parser.parse_args()
    try:
        if args.command == 'init':
            print('Pin this public root fingerprint independently:', CommunicationJournal.initialize(args.service_dir))
            return 0
        if not args.trusted_root: raise ValueError('Root fingerprint required')
        journal = CommunicationJournal(args.service_dir, args.trusted_root)
        if args.command in ('add-user', 'set-scope'):
            scope = {k: getattr(args,k) for k in ('agents','sessions','kinds','classifications')}
            if args.command == 'add-user': journal.add_user(args.user, _passphrase(args.user), scope)
            else: journal.set_scope(args.user, scope)
        elif args.command == 'rotate-password': journal.rotate_password(args.user, _passphrase(args.user), reactivate=args.reactivate)
        elif args.command == 'revoke-user': journal.revoke_user(args.user)
        elif args.command == 'rotate-keys': print('Current epoch:', journal.rotate_keys())
        elif args.command == 'revoke-epoch': journal.revoke_epoch(args.epoch)
        elif args.command == 'status': print(json.dumps(journal.status(), indent=2, ensure_ascii=False))
        elif args.command == 'view':
            print(json.dumps(journal.view(args.user, getpass.getpass('Passphrase: '), agent=args.agent,
                                          session=args.session, kind=args.kind, limit=args.limit, after=args.after), indent=2, ensure_ascii=False))
        elif args.command == 'audit': print(json.dumps(journal.read_audit(limit=args.limit, after=args.after), indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Communication operation failed ({type(exc).__name__}). Check credentials, permissions and inputs; inspect status/audit before retrying administrative changes.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
