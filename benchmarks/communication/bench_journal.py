"""Measure native SLDC Communication operations in a fresh temporary service."""
import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cryptography

from src.communication.communication_storage import CommunicationJournal
from src.communication.communication_types import new_message


def benchmark(count):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / 'service'
        pin = CommunicationJournal.initialize(root)
        journal = CommunicationJournal(root, pin)
        password = 'temporary-benchmark-passphrase'
        journal.add_user('reader', password, {key: ['*'] for key in
                                              ('agents', 'sessions', 'kinds', 'classifications')})
        samples = {}
        def measure(name, operation):
            start = time.perf_counter_ns()
            operation()
            samples.setdefault(name, []).append((time.perf_counter_ns() - start) / 1e6)

        for _ in range(count):
            record = new_message('producer', 'consumer', 'event', {'sample': 'x' * 256},
                                 session_id='benchmark')
            measure('append', lambda: journal.append(record))
        for _ in range(count):
            measure('view_10', lambda: journal.view('reader', password, limit=10))
        return {'python': sys.version.split()[0], 'cryptography': cryptography.__version__,
                'platform': platform.platform(), 'cpu_count': os.cpu_count(),
                'records': count, 'bytes_on_disk_logical': sum(
                    path.stat().st_size for path in root.rglob('*') if path.is_file()),
                'median_ms': {name: statistics.median(times) for name, times in samples.items()},
                'samples_ms': samples}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=int, default=25)
    args = parser.parse_args()
    if args.records < 1:
        parser.error('--records must be positive')
    print(json.dumps(benchmark(args.records), indent=2))
