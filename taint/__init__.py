"""TAINT — prompt-injection provenance detection (defensive, offline scaffold).

TAINT detects, per agent response/action, whether an LLM/agent obeyed *untrusted
injected content* instead of the legitimate trusted instruction, and applies an
elevation policy (allow/flag/block) to cut successful injections.

This package runs **fully offline** on a deterministic mock agent. All injections
are **benign markers** (e.g. appending the token ``PWNED``); no harmful content is
ever produced or executed. See the README safety note.
"""

from __future__ import annotations

__version__ = "0.5.0"  # ~50% scaffold

__all__ = [
    "agent",
    "harness",
    "dataset",
    "detect_canary",
    "detect_judge",
    "detect_probe",
    "policy",
    "eval",
]
