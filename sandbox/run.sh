#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SANDBOX_NAME="claude-$(basename "$REPOROOT")"

docker sandbox create \
  --name "$SANDBOX_NAME" \
  --template uv-claude-sandbox:latest \
  --load-local-template \
  claude "$REPOROOT" 2>/dev/null || true

if (($#)); then
  docker sandbox run "$SANDBOX_NAME" -- "$@"
else
  docker sandbox run "$SANDBOX_NAME"
fi