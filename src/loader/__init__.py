from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
__path__.append(str(_ROOT / "dataloaders" / "core"))
__path__.append(str(_ROOT))
