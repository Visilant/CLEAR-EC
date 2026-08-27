#!/usr/bin/env bash
# Thin wrapper: the fair overnight runner is run_overnight.py.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
exec "${CODE_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/run_overnight.py" "$@"
