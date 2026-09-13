"""Read-only queue and validation summary.  Optional argument: run root."""
import json
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[4] / (sys.argv[1] if len(sys.argv) > 1 else 'runs/ovcr_adapter_padapter_v2')
status = root/'suite_status.json'
print(status.read_text() if status.exists() else 'No suite launched yet')
for path in sorted(root.glob('*/model/validation.jsonl')):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if rows: print(path.parent.parent.name, json.dumps(rows[-1]))
