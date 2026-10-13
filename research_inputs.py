"""Resolve immutable research input hashes after daily source data advances."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def research_input(name, expected_hash):
    if Path(name).name != name:
        raise ValueError("Expected a repository-root input name")
    choices = [ROOT / name] + sorted((ROOT / "research_snapshots").glob("*/" + name))
    for path in choices:
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash:
            return path
    raise ValueError("No preserved input matches the research fingerprint: " + name)
