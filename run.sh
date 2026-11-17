#!/usr/bin/env bash
set -eu

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" my_main.py "$@"
