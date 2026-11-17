#!/usr/bin/env python3
"""Check the code-only candidate for accidental assets and identifying text.

This bounded static check does not establish consent, rights or anonymity.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "build", "dist"}
ASSET_SUFFIXES = {
    ".pt", ".pth", ".onnx", ".safetensors", ".pkl", ".npy", ".npz",
    ".mp4", ".avi", ".mov", ".mkv", ".zip", ".csv", ".parquet",
    ".png", ".jpg", ".jpeg", ".obj", ".stl", ".step", ".blend",
}
TEXT_SUFFIXES = {".py", ".json", ".toml", ".md", ".txt", ".yml", ".yaml"}
PATTERNS = {
    "access-token shape": re.compile(r"\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{20,}"),
    "AWS access-key shape": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private-key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "user-specific absolute path": re.compile(
        "/" + r"home/[^/\s\"'<>]+|[A-Za-z]:[\\/]Users[\\/][^\\/\s\"'<>]+"
    ),
    "private-network address": re.compile(
        r"(?<![\w.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\w.])"
    ),
    "private questionnaire crosswalk": re.compile("FORM_ID" + "_TO_SUBJECT"),
    "team-storage link": re.compile(r"https?://[^\s/]*sharepoint\.com/", re.I),
}


def main() -> int:
    problems = []
    checked = 0
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            problems.append(f"{relative}: symlink not allowed in the release bundle")
            continue
        if not path.is_file():
            continue
        checked += 1
        if path.suffix.lower() in ASSET_SUFFIXES:
            problems.append(f"{relative}: data/model/media asset in code-only tree")
        if any(part in {"experiments", "results", "runs", ".cache", ".aws"} for part in relative.parts):
            problems.append(f"{relative}: generated/private data directory")
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
            if path.suffix == ".py":
                ast.parse(content, filename=str(relative))
            elif path.suffix == ".json":
                json.loads(content)
        except (UnicodeError, ValueError, SyntaxError) as error:
            problems.append(f"{relative}: invalid text, Python or JSON ({type(error).__name__})")
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(content):
                line = content.count("\n", 0, match.start()) + 1
                # Report location and kind, without echoing potential secrets.
                problems.append(f"{relative}:{line}: {label}")
    if problems:
        print("Release-content check failed:")
        print("\n".join(f"- {problem}" for problem in problems))
        return 1
    print(f"Release-content check passed: {checked} files; no forbidden assets or configured text matches.")
    print("Third-party rights, consent and missing external assets remain separate release requirements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
