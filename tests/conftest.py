"""Make ``scripts/triage`` importable from the tests without installing it."""

import sys
from pathlib import Path

TRIAGE_DIR = Path(__file__).resolve().parent.parent / "scripts" / "triage"
if str(TRIAGE_DIR) not in sys.path:
    sys.path.insert(0, str(TRIAGE_DIR))
