#!/usr/bin/env bash
# Training-arm runner for the encoder scaling replication
# (dev/AUDIT_2026-09-08.md finding 7 + recommendation (c)):
#
#   v2_788   -- runs/encoder_gold_v2.jsonl, n_train=788   (today's data, today's size)
#   v3_788   -- runs/encoder_gold_v3.jsonl, n_train=788   (same size as v2_788, CLEANER data only)
#   v3_3000  -- runs/encoder_gold_v3.jsonl, n_train=3000  (v3_788 + MORE data)
#
# Comparing v2_788 vs v3_788 isolates data quality (top-1 prune + richer
# extraction) at equal size and equal optimizer-step budget; v3_788 vs
# v3_3000 isolates corpus size at equal quality. Every arm x seed trains for
# exactly --max-steps optimizer steps (scripts/train_encoder.py item A) on a
# stratified subset of its gold pool (item B) that EXCLUDES the shared
# held-out sentences (item C, scripts/make_holdout.py), and is scored on
# those same held-out sentences against BOTH v2 and v3 targets.
#
# Usage:
#   bash scripts/run_encoder_arms.sh                # sequential, all 6 (arm x seed) runs
#   bash scripts/run_encoder_arms.sh --parallel 2    # up to 2 runs at a time
#   bash scripts/run_encoder_arms.sh --dry-run       # print the 6 commands and exit, no training
#   STEPS=20000 bash scripts/run_encoder_arms.sh     # override the optimizer-step budget
#   GOLD_V3=runs/encoder_gold_v3_draft.jsonl bash scripts/run_encoder_arms.sh
#
# Do NOT run a full arm without the lead's go-ahead -- at ~40,000 steps this
# is many CPU-hours per arm x seed (see the printed wall-clock estimate,
# which is projected from a quick --smoke throughput calibration run first).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STEPS="${STEPS:-40000}"
MAX_SECONDS="${MAX_SECONDS:-172800}"   # 48h hard per-run ceiling; --max-steps is the real budget
GOLD_V2="${GOLD_V2:-runs/encoder_gold_v2.jsonl}"
GOLD_V3="${GOLD_V3:-runs/encoder_gold_v3.jsonl}"
HOLDOUT_FILE="${HOLDOUT_FILE:-runs/holdout_sentences.txt}"
SEEDS="${SEEDS:-0 1}"
BEAM_WIDTH="${BEAM_WIDTH:-6}"
K="${K:-6}"
SMOKE_CALIB_STEPS="${SMOKE_CALIB_STEPS:-30}"

PARALLEL=1
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel)
      PARALLEL="$2"; shift 2 ;;
    --parallel=*)
      PARALLEL="${1#*=}"; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    *)
      echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

ARMS_DIR="runs/arms"
mkdir -p "$ARMS_DIR"
SUMMARY="$ARMS_DIR/summary.tsv"
if [[ ! -f "$SUMMARY" ]]; then
  printf 'arm\tseed\tgold\tn_train\toptimizer_steps\tstop_reason\tepoch_fraction\ttrain_wall_s\tholdout_v2_edge_precision\tholdout_v2_edge_recall\tholdout_v2_rank1_edge_f1\tholdout_v2_mean_forest_width\tholdout_v3_edge_precision\tholdout_v3_rank1_edge_f1\tcheckpoint\n' > "$SUMMARY"
fi

if [[ ! -f "$HOLDOUT_FILE" ]]; then
  echo "ERROR: holdout file $HOLDOUT_FILE not found. Run: python scripts/make_holdout.py" >&2
  exit 1
fi

V3_AVAILABLE=1
if [[ ! -f "$GOLD_V3" ]]; then
  V3_AVAILABLE=0
fi

# arm_name:gold_file:n_train
ARM_DEFS=(
  "v2_788:${GOLD_V2}:788"
  "v3_788:${GOLD_V3}:788"
  "v3_3000:${GOLD_V3}:3000"
)

# ARMS restricts which arm names from ARM_DEFS are considered at all (space-
# separated), default all three -- e.g. ARMS="v2_788 v3_788" to skip v3_3000
# outright instead of relying on the GOLD_V3-missing skip path.
ARMS="${ARMS:-v2_788 v3_788 v3_3000}"
_arms_filtered=()
for arm_def in "${ARM_DEFS[@]}"; do
  IFS=':' read -r arm _ _ <<< "$arm_def"
  for wanted in $ARMS; do
    if [[ "$arm" == "$wanted" ]]; then _arms_filtered+=("$arm_def"); break; fi
  done
done
ARM_DEFS=("${_arms_filtered[@]}")

build_cmd() {
  local arm="$1" gold="$2" n_train="$3" seed="$4"
  local out="$ARMS_DIR/${arm}_${seed}.pt"
  local log="$ARMS_DIR/${arm}_${seed}.log"
  local extra_eval=""
  if [[ "$gold" == "$GOLD_V2" ]]; then
    extra_eval="--eval-gold ${GOLD_V2}"
    if [[ "$V3_AVAILABLE" == "1" ]]; then extra_eval="$extra_eval --eval-gold-alt ${GOLD_V3}"; fi
  else
    extra_eval="--eval-gold ${GOLD_V2} --eval-gold-alt ${GOLD_V3}"
  fi
  echo "python scripts/train_encoder.py --gold ${gold} --n-train ${n_train}" \
       "--seed ${seed} --subset-seed ${seed} --max-steps ${STEPS} --max-seconds ${MAX_SECONDS}" \
       "--holdout-file ${HOLDOUT_FILE} ${extra_eval}" \
       "--beam-width ${BEAM_WIDTH} --k ${K} --out ${out}"
}

echo "=== encoder training arms ==="
echo "STEPS=$STEPS  MAX_SECONDS=$MAX_SECONDS  PARALLEL=$PARALLEL  ARMS=$ARMS"
echo "GOLD_V2=$GOLD_V2  GOLD_V3=$GOLD_V3 (available=$V3_AVAILABLE)"
echo "HOLDOUT_FILE=$HOLDOUT_FILE"
echo

# Build all 6 (arm x seed) commands unconditionally -- shown in full even
# when GOLD_V3 is missing, so --dry-run documents the whole plan. Only the
# RUNNABLE subset (v3 arms dropped if GOLD_V3 is absent) is actually
# executed below.
ALL_COMMANDS=()
RUNNABLE_COMMANDS=()
V3_SKIPPED_ARMS=()
for arm_def in "${ARM_DEFS[@]}"; do
  IFS=':' read -r arm gold n_train <<< "$arm_def"
  needs_v3=0
  if [[ "$gold" == "$GOLD_V3" ]]; then needs_v3=1; fi
  for seed in $SEEDS; do
    cmd="$(build_cmd "$arm" "$gold" "$n_train" "$seed")"
    entry="$arm|$seed|$gold|$n_train|$cmd"
    ALL_COMMANDS+=("$entry")
    if [[ "$needs_v3" == "1" && "$V3_AVAILABLE" == "0" ]]; then
      : # not runnable yet
    else
      RUNNABLE_COMMANDS+=("$entry")
    fi
  done
  if [[ "$needs_v3" == "1" && "$V3_AVAILABLE" == "0" ]]; then
    V3_SKIPPED_ARMS+=("$arm")
  fi
done

if [[ "$V3_AVAILABLE" == "0" ]]; then
  echo "SKIP: arms ${V3_SKIPPED_ARMS[*]} need $GOLD_V3, which does not exist yet -- run the v3 gold build first. Commands are still listed below for reference; they will not run."
fi

echo "Planned commands (${#ALL_COMMANDS[@]} total, ${#RUNNABLE_COMMANDS[@]} runnable now):"
for entry in "${ALL_COMMANDS[@]}"; do
  IFS='|' read -r arm seed gold n_train cmd <<< "$entry"
  tag=""
  if [[ "$gold" == "$GOLD_V3" && "$V3_AVAILABLE" == "0" ]]; then tag=" [SKIP: $GOLD_V3 missing]"; fi
  echo "  [$arm seed=$seed]$tag $cmd"
done
echo

COMMANDS=("${RUNNABLE_COMMANDS[@]}")
if [[ "${#COMMANDS[@]}" -eq 0 ]]; then
  echo "No arms to run (is $GOLD_V3 missing?)."
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "--dry-run: not running anything."
  exit 0
fi

# Quick throughput calibration (a short --smoke run) so we can print a
# wall-clock ESTIMATE per arm before committing CPU-hours. Smoke dims
# (d_model=64) are smaller than the real arms (d_model=128), so treat this
# as a lower-bound proxy -- the FINAL MESSAGE this session reports also
# carries a real-dims measurement taken separately.
echo "=== throughput calibration: --smoke --max-steps ${SMOKE_CALIB_STEPS} ==="
CALIB_LOG="$ARMS_DIR/_smoke_calib.log"
nice -n 10 python scripts/train_encoder.py --smoke --max-steps "$SMOKE_CALIB_STEPS" \
  --holdout-file "$HOLDOUT_FILE" --gold "$GOLD_V2" --out "$ARMS_DIR/_smoke_calib.pt" \
  > "$CALIB_LOG" 2>&1
S_PER_STEP="$(grep -oE 's/step=[0-9.]+' "$CALIB_LOG" | tail -1 | sed -E 's/[^0-9.]*([0-9.]+).*/\1/')"
if [[ -z "${S_PER_STEP:-}" ]]; then
  S_PER_STEP="$(grep -oE '\(([0-9.]+) s/step\)' "$CALIB_LOG" | tail -1 | grep -oE '[0-9.]+')"
fi
if [[ -n "${S_PER_STEP:-}" ]]; then
  EST_S=$(python3 -c "print(${S_PER_STEP} * ${STEPS})")
  EST_H=$(python3 -c "print(round(${S_PER_STEP} * ${STEPS} / 3600, 2))")
  echo "measured smoke throughput: ${S_PER_STEP} s/step -> estimated ${EST_S}s (${EST_H}h) per arm at ${STEPS} steps"
  echo "(smoke dims d_model=64; real arms use d_model=128 and will be slower -- see the session's real-dims calibration)"
else
  echo "WARNING: could not parse s/step from $CALIB_LOG; skipping wall-clock estimate"
fi
echo

run_one() {
  local entry="$1"
  IFS='|' read -r arm seed gold n_train cmd <<< "$entry"
  local log="$ARMS_DIR/${arm}_${seed}.log"
  local out="$ARMS_DIR/${arm}_${seed}.pt"
  echo "[$arm seed=$seed] starting -> $log"
  nohup nice -n 10 bash -c "$cmd" > "$log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$arm seed=$seed] FAILED (exit $rc) -- see $log" >&2
    return
  fi
  python3 - "$arm" "$seed" "$gold" "$n_train" "$out" "$SUMMARY" <<'PYEOF'
import sys, torch
arm, seed, gold, n_train, out, summary = sys.argv[1:7]
ckpt = torch.load(out, map_location="cpu", weights_only=False)
cfg = ckpt["config"]
m = ckpt["metrics"].get("holdout", {})
alt = ckpt["metrics"].get("holdout_alt", {})
row = [
    arm, seed, gold, n_train,
    str(cfg.get("optimizer_steps")), str(cfg.get("stop_reason")),
    f"{cfg.get('epoch_fraction', float('nan')):.4f}",
    f"{ckpt.get('train_wallclock_s', float('nan')):.1f}",
    f"{m.get('edge_precision', float('nan')):.4f}", f"{m.get('edge_recall', float('nan')):.4f}",
    f"{m.get('rank1_edge_f1', float('nan')):.4f}", f"{m.get('mean_forest_width', float('nan')):.2f}",
    f"{alt.get('edge_precision', float('nan')):.4f}", f"{alt.get('rank1_edge_f1', float('nan')):.4f}",
    out,
]
with open(summary, "a") as f:
    f.write("\t".join(row) + "\n")
PYEOF
  echo "[$arm seed=$seed] done -> $SUMMARY"
}
export -f run_one
export ARMS_DIR SUMMARY

if [[ "$PARALLEL" -le 1 ]]; then
  for entry in "${COMMANDS[@]}"; do
    run_one "$entry"
  done
else
  printf '%s\n' "${COMMANDS[@]}" | xargs -I{} -P "$PARALLEL" bash -c 'run_one "$@"' _ {}
fi

echo
echo "All arms done. Summary: $SUMMARY"
