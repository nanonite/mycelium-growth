#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$REPO/work_dirs/segformer-b0-repro/full-release-all-labeled-20261009"
LOG="$WORK/run.log"
STATE="$WORK/state.json"
cd "$REPO"
mkdir -p "$WORK"

while true; do
  printf '\n=== starting a fresh GPU inference process at %s ===\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"

  set +e
  nix develop .#gpu --command timeout --signal=INT --kill-after=60s 20m \
    env PYTHONUNBUFFERED=1 PYTHONPATH=scripts/segformer_shim \
    python scripts/segformer_full_release.py run 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e

  status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$STATE")"
  printf 'batch ended: exit=%s state=%s\n' "$rc" "$status" | tee -a "$LOG"

  case "$status" in
    complete) exit 0 ;;
    completed_with_errors) exit 2 ;;
    running) ;;
    *) printf 'unexpected run state: %s\n' "$status" >&2; exit 1 ;;
  esac

  if [[ "$rc" != 124 && "$rc" != 130 && "$rc" != 137 ]]; then
    printf 'inference exited before its 20-minute batch limit (exit=%s)\n' "$rc" >&2
    exit "$rc"
  fi
done
