"""Frontend for iHOS App."""
from pathlib import Path


def where() -> Path:
    """Return path to the frontend."""
    return Path(__file__).parent
