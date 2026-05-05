#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

docker build \
  -f "$SCRIPT_DIR/Dockerfile.sandbox" \
  -t uv-claude-sandbox:latest \
  "$REPOROOT"