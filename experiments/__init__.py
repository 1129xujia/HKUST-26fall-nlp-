"""Research entry points; shared runtime code lives in src/."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def raw_data_dir():
    """Locate the shared official distribution without copying its data."""
    root = ROOT / "data/raw/multidoc2dial"
    candidates = sorted(
        (p for p in root.rglob("multidoc2dial_doc.json") if "__MACOSX" not in p.parts),
        key=lambda p: (len(p.parts), p.as_posix()),
    )
    return candidates[0].parent if candidates else root
