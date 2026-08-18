"""Fail when the Git index contains private paths or generated artifacts."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_PATH_PARTS = {"outputs", ".local", "cache"}
FORBIDDEN_SUFFIXES = {".nc", ".npy", ".npz", ".pt", ".pth", ".log"}
PRIVATE_PATTERNS = {
    "server mount": re.compile(r"/mnt/[A-Za-z0-9_.-]+"),
    "private home": re.compile(r"/home/[A-Za-z0-9_.-]+"),
    "Windows user path": re.compile(r"[A-Za-z]:[/\\]Users[/\\][^/\\]+", re.IGNORECASE),
    "private IPv4": re.compile(
        r"(?<!\d)(?:10\.(?:\d{1,3}\.){2}\d{1,3}|"
        r"192\.168\.(?:\d{1,3}\.)\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3})(?!\d)"
    ),
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / value.decode("utf-8") for value in result.stdout.split(b"\0") if value]


def audit(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        relative = path.relative_to(ROOT)
        if FORBIDDEN_PATH_PARTS.intersection(relative.parts):
            findings.append(f"generated/private directory is tracked: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"generated artifact is tracked: {relative}")
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label} found in {relative}")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    findings = audit(tracked_files())
    if findings:
        raise SystemExit("\n".join(f"[public-audit] {item}" for item in findings))
    print("[public-audit] tracked tree is clean")


if __name__ == "__main__":
    main()
