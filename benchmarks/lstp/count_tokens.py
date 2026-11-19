"""Optional local tokenizer process. Install tiktoken separately; specify the encoding.

Counts instruction and payload content. API framing, cached billing, generated
output and reasoning are measured only by a live provider, not this counter.
"""

import argparse
import json
import sys

import tiktoken

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--encoding", required=True)
args = parser.parse_args()
encoding = tiktoken.get_encoding(args.encoding)
for line in sys.stdin:
    item = json.loads(line)
    print(
        len(encoding.encode(item["system"], disallowed_special=()))
        + len(encoding.encode(item["user"], disallowed_special=()))
    )
