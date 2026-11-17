#!/usr/bin/env bash
set -eu

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
case "${1:-}" in
    ablation|design-choice|hyper) entry=main_dc.py ;;
    *) entry=my_main.py ;;
esac
exec "${PYTHON:-python}" "$entry" "$@"
