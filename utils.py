"""Shared subprocess launcher for the PAVE entry points."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run_module(module, options):
    return subprocess.run([sys.executable, "-m", module, *options], cwd=ROOT, check=False).returncode
