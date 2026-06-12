#!/usr/bin/env bash
# submit_all.sh — bulk-submit the burgers_ab frontier sweep TOMLs.
#
# Run from the q8020 workspace root, so that q8020-sweep resolves the
# TOMLs' relative paths (./q8020-mps-burgers/.venv, ./q8020-mps-burgers/src/...).
#
# Usage:
#   ./q8020-mps-burgers/input/frontier/submit_all.sh [options]
#
# Options:
#   --tier {q6|q7|q8|all}   Limit to one q-tier (default: all)
#   --file NAME             Submit just one TOML by basename (e.g. burgers_ab_shock_q6.toml)
#   --dry-run               Pass --dry-run to q8020-sweep (no sbatch submission)
#   --no-pause              Skip interactive confirms between tiers
#   -h, --help              Print this help

set -euo pipefail

TOML_DIR="q8020-mps-burgers/input/frontier"
SWEEP_VENV="${SWEEP_VENV:-$HOME/venvs/q8020-sweep}"
TIER="all"
DRY_RUN=""
NO_PAUSE=0
SINGLE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)     TIER="$2"; shift 2;;
    --file)     SINGLE="$2"; shift 2;;
    --dry-run)  DRY_RUN="--dry-run"; shift;;
    --no-pause) NO_PAUSE=1; shift;;
    -h|--help)  sed -n '2,15p' "$0"; exit 0;;
    *) echo "unknown arg: $1 (use -h for help)" >&2; exit 1;;
  esac
done

if [[ ! -d "$TOML_DIR" ]]; then
  echo "ERROR: $TOML_DIR not found." >&2
  echo "Run this script from the q8020 workspace root, e.g.:" >&2
  echo "  cd ~/q8020 && ./q8020-mps-burgers/input/frontier/submit_all.sh" >&2
  exit 1
fi

# Activate the sweep venv (where q8020-sweep is installed) if not already on PATH.
# Override with: SWEEP_VENV=/path/to/venv ./submit_all.sh
if ! command -v q8020-sweep >/dev/null 2>&1; then
  if [[ -f "$SWEEP_VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$SWEEP_VENV/bin/activate"
  else
    echo "ERROR: q8020-sweep not on PATH and sweep venv not found at $SWEEP_VENV." >&2
    echo "Either install q8020-cfd-metautil into a venv at \$HOME/venvs/q8020-sweep," >&2
    echo "or override with: SWEEP_VENV=/path/to/venv $0" >&2
    exit 1
  fi
  if ! command -v q8020-sweep >/dev/null 2>&1; then
    echo "ERROR: activated $SWEEP_VENV but q8020-sweep still not on PATH." >&2
    exit 1
  fi
fi

submit_one() {
  local toml="$1"
  if [[ ! -f "$toml" ]]; then
    echo "  SKIP: $toml not found"
    return
  fi
  echo
  echo "=================================================="
  echo "Submitting: $toml"
  echo "=================================================="
  q8020-sweep "$toml" $DRY_RUN
}

submit_tier() {
  local glob="$1"
  local label="$2"
  echo
  echo "######  TIER: $label  ######"
  shopt -s nullglob
  local files=( $TOML_DIR/$glob )
  shopt -u nullglob
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "  (no files matched $glob)"
    return
  fi
  for toml in "${files[@]}"; do
    submit_one "$toml"
  done
}

confirm() {
  [[ $NO_PAUSE -eq 1 ]] && return
  local yn
  if read -r -p "Proceed to next tier? [y/N] " yn; then
    case "$yn" in [yY]*) return;; *) echo "Stopping."; exit 0;; esac
  else
    echo "Stopping (EOF)."; exit 0
  fi
}

if [[ -n "$SINGLE" ]]; then
  submit_one "$TOML_DIR/$SINGLE"
  exit 0
fi

case "$TIER" in
  q6)  submit_tier "burgers_ab_*q6*.toml" "q6 (batch, 2h)";;
  q7)  submit_tier "burgers_ab_*q7*.toml" "q7";;
  q8)  submit_tier "burgers_ab_*q8*.toml" "q8 (extended, 24h)";;
  all)
    submit_tier "burgers_ab_*q6*.toml" "q6 (batch, 2h)"
    confirm
    submit_tier "burgers_ab_*q7*.toml" "q7"
    confirm
    submit_tier "burgers_ab_*q8*.toml" "q8 (extended, 24h)"
    ;;
  *) echo "unknown tier: $TIER (use q6|q7|q8|all)" >&2; exit 1;;
esac

echo
echo "All requested submissions issued."
echo "Check queue: squeue -u \$USER"
echo "Job logs:    ls -lt q8020-mps-burgers/input/frontier/../../*/sweep_*"
