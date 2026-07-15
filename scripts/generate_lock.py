"""Regenerate requirements-lock.txt from the running interpreter's environment.

Invoked by `make freeze`. Writing the file from Python (not shell `echo`
chains) keeps this correct on both POSIX shells and cmd.exe — cmd.exe's
`echo` is a built-in with its own quoting rules and doesn't strip the
surrounding quotes the way sh/bash does, so a Makefile recipe built out of
`echo "..." >> file` lines silently corrupts the file on Windows.

Usage:
    .venv/bin/python scripts/generate_lock.py
"""
from __future__ import annotations

import subprocess
import sys

HEADER = """\
# =============================================================================
# requirements-lock.txt — full dependency closure, direct + transitive.
#
# Generated via `make freeze` (= `pip freeze` from a working .venv). This is
# what `make install` actually installs from, for exact reproducibility
# across machines/installer builds. Do not hand-edit — regenerate with
# `make freeze` after `make install` picks up a change to requirements.txt,
# then commit both files together.
# =============================================================================

--extra-index-url https://download.pytorch.org/whl/cu126

"""


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        capture_output=True,
        text=True,
        check=True,
    )
    with open("requirements-lock.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER)
        f.write(result.stdout)
    print(f"requirements-lock.txt regenerated from {sys.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
