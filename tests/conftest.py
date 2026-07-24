"""Make the project importable when pytest collects from tests/.

With pytest's default import mode, the directory inserted into sys.path is the
first one without an __init__.py — that is tests/, not the repository root — so
`from core... import ...` would fail. Inserting the root here keeps the test
files free of sys.path boilerplate.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
