#!/usr/bin/env bash
set -euo pipefail
PITPO_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PITPO_SCRIPT_DIR/runtime_env.sh"
exec "$@"
