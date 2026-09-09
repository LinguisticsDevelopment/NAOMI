#!/usr/bin/env bash
# Training-arm runner for the encoder scaling replication
# (dev/AUDIT_2026-09-08.md finding 7 + recommendation (c);
# encoder-train-arms-v2 adds periodic dev eval + keep-best selection so
# arms with different gold-derivations-per-record aren't compared at a
# fixed optimizer-step budget that is secretly a very different number of
# epochs -- see scripts/train_encoder.py's --eval-every/--keep-best):
#
#   v2_788        -- runs/encoder_gold_v2.jsonl, n_train=788   (today's data, today's size)
#   v3_788        -- runs/encoder_gold_v3.jsonl, n_train=788   (same size as v2_788, CLEANER data only)
#   v3_3000       -- runs/encoder_gold_v3.jsonl, n_train=3000  (v3_788 + MORE data)
#   v4b_788       -- runs/encoder_gold_v4b.jsonl, n_train=788  (v4b gold, same size as v2_788)
#   v4b_788_hard  -- runs/encoder_gold_v4b.jsonl + runs/hard_gold_train.jsonl (v3, 1038 records,
#                    ALL of it via --n-train-per-file) + 788 real v4b records (~1.3:1 hard:real)
#                    (v4b gold topped up with the FULL hard-construction gold; scored on its own
#                    held-out hard-gold v3 test splits via --extra-eval, reported per family)
#   v4b_788_hardsmall -- runs/encoder_gold_v4b.jsonl + runs/hard_gold_train_small.jsonl (v3,
#                    200-record family-stratified sample, ALL of it via --n-train-per-file) +
#                    788 real v4b records (1:4 hard:real) -- same eval wiring as v4b_788_hard,
#                    measures whether a lighter hard-gold mix preserves in-domain (v4b/v2) F1
#                    while still improving the widened families on unseen (test_template) forms.
#                    arms-3(b): both hard arms use --eval-gold v4b (not v2) so --keep-best
#                    selects on v4b targets (the reference number this study compares against),
#                    and --eval-gold-alt v2 for continuity with earlier arms.
#
# Comparing v2_788 vs v3_788 isolates data quality (top-1 prune + richer
# extraction) at equal size and equal optimizer-step budget; v3_788 vs
# v3_3000 isolates corpus size at equal quality. Every arm x seed trains for
# exactly --max-steps optimizer steps (scripts/train_encoder.py item A) on a
# stratified subset of its gold pool (item B) that EXCLUDES the shared
# held-out sentences (item C, scripts/make_holdout.py) AND the shared dev
# holdout sentences (--dev-holdout-file, whenever EVAL_EVERY>0), and is
# scored on the test holdout regardless of which gold file trains it
# (--eval-gold / --eval-gold-alt).
#
# Every arm also gets a periodic dev eval (--eval-every) and keep-best
# checkpoint selection (--keep-best) by default: --max-steps is the same
# for every arm, but a single-tree gold record trains many fewer
# derivations than a forest-gold record, so the SAME step budget is a very
# different number of epochs over the corpus -- --keep-best scores each arm
# at ITS OWN best point on the dev holdout instead of wherever --max-steps
# happens to land, and epochs_equivalent (scripts/train_encoder.py item C)
# makes that epoch count visible in summary.tsv.
#
# Usage:
#   bash scripts/run_encoder_arms.sh                # sequential, all (arm x seed) runs
#   bash scripts/run_encoder_arms.sh --parallel 2    # up to 2 runs at a time
#   bash scripts/run_encoder_arms.sh --dry-run       # print the planned commands and exit, no training
#   STEPS=20000 bash scripts/run_encoder_arms.sh     # override the optimizer-step budget
#   ARMS="v2_788 v4b_788" bash scripts/run_encoder_arms.sh   # only run these arms (space/comma list)
#   EVAL_EVERY=0 KEEP_BEST=0 bash scripts/run_encoder_arms.sh  # opt back out of encoder-train-arms-v2
#   GOLD_V3=runs/encoder_gold_v3_draft.jsonl bash scripts/run_encoder_arms.sh
#   METRIC=edge bash scripts/run_encoder_arms.sh    # opt back out of the USVS-graded columns
#   LOSS=usvs-soft bash scripts/run_encoder_arms.sh # train with the graded soft CE targets
#
# Do NOT run a full arm without the lead's go-ahead -- at tens of thousands
# of steps this is many CPU-hours per arm x seed (see the printed wall-clock
# estimate, which is projected from a quick --smoke throughput calibration
# run first).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STEPS="${STEPS:-40000}"
MAX_SECONDS="${MAX_SECONDS:-172800}"   # 48h hard per-run ceiling; --max-steps is the real budget
GOLD_V2="${GOLD_V2:-runs/encoder_gold_v2.jsonl}"
GOLD_V3="${GOLD_V3:-runs/encoder_gold_v3.jsonl}"
GOLD_V4B="${GOLD_V4B:-runs/encoder_gold_v4b.jsonl}"
GOLD_V4B_MARGIN="${GOLD_V4B_MARGIN:-runs/encoder_gold_v4b_margin.jsonl}"
GOLD_V4B_ALL="${GOLD_V4B_ALL:-runs/encoder_gold_v4b_all.jsonl}"
HARD_GOLD_TRAIN="${HARD_GOLD_TRAIN:-runs/hard_gold_train.jsonl}"
HARD_GOLD_TRAIN_SMALL="${HARD_GOLD_TRAIN_SMALL:-runs/hard_gold_train_small.jsonl}"
HARD_GOLD_TEST_FILLER="${HARD_GOLD_TEST_FILLER:-runs/hard_gold_test_filler.jsonl}"
HARD_GOLD_TEST_TEMPLATE="${HARD_GOLD_TEST_TEMPLATE:-runs/hard_gold_test_template.jsonl}"
HOLDOUT_FILE="${HOLDOUT_FILE:-runs/holdout_sentences.txt}"
HOLDOUT_DEV_FILE="${HOLDOUT_DEV_FILE:-runs/holdout_dev_sentences.txt}"
SEEDS="${SEEDS:-0 1}"
BEAM_WIDTH="${BEAM_WIDTH:-6}"
K="${K:-6}"
SMOKE_CALIB_STEPS="${SMOKE_CALIB_STEPS:-30}"
EVAL_EVERY="${EVAL_EVERY:-250}"
KEEP_BEST="${KEEP_BEST:-1}"
ARMS="${ARMS:-}"   # space/comma list of arm names to run; empty = all

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
  printf 'arm\tseed\tgold\tn_train\toptimizer_steps\tstop_reason\tepoch_fraction\ttrain_wall_s\tholdout_v2_edge_precision\tholdout_v2_edge_recall\tholdout_v2_rank1_edge_f1\tholdout_v2_mean_forest_width\tholdout_alt_edge_precision\tholdout_alt_rank1_edge_f1\tbest_step\tbest_dev_rank1_f1\tlast_dev_rank1_f1\tepochs_equivalent\textra_filler_rank1_f1_json\textra_template_rank1_f1_json\tcheckpoint\n' > "$SUMMARY"
fi

if [[ ! -f "$HOLDOUT_FILE" ]]; then
  echo "ERROR: holdout file $HOLDOUT_FILE not found. Run: python scripts/make_holdout.py" >&2
  exit 1
fi
if [[ "$EVAL_EVERY" -gt 0 && ! -f "$HOLDOUT_DEV_FILE" ]]; then
  echo "ERROR: dev holdout file $HOLDOUT_DEV_FILE not found (needed because EVAL_EVERY=$EVAL_EVERY > 0). Run: python scripts/make_holdout.py" >&2
  exit 1
fi

V3_AVAILABLE=1
if [[ ! -f "$GOLD_V3" ]]; then V3_AVAILABLE=0; fi
V4B_AVAILABLE=1
if [[ ! -f "$GOLD_V4B" ]]; then V4B_AVAILABLE=0; fi
V4B_MARGIN_AVAILABLE=1
if [[ ! -f "$GOLD_V4B_MARGIN" ]]; then V4B_MARGIN_AVAILABLE=0; fi
V4B_ALL_AVAILABLE=1
if [[ ! -f "$GOLD_V4B_ALL" ]]; then V4B_ALL_AVAILABLE=0; fi
HARD_AVAILABLE=1
if [[ ! -f "$HARD_GOLD_TRAIN" ]]; then HARD_AVAILABLE=0; fi
HARD_SMALL_AVAILABLE=1
if [[ ! -f "$HARD_GOLD_TRAIN_SMALL" ]]; then HARD_SMALL_AVAILABLE=0; fi

# arm_name:gold_file:n_train
ARM_DEFS=(
  "v2_788:${GOLD_V2}:788"
  "v3_788:${GOLD_V3}:788"
  "v3_3000:${GOLD_V3}:3000"
  "v4b_788:${GOLD_V4B}:788"
  "v4b_788_hard:${GOLD_V4B},${HARD_GOLD_TRAIN}:788"
  "v4b_788_hardsmall:${GOLD_V4B},${HARD_GOLD_TRAIN_SMALL}:788"
  "v4b_margin_788:${GOLD_V4B_MARGIN}:788"
  "v4b_all_788:${GOLD_V4B_ALL}:788"
)

# Whether all the gold files an arm needs are actually present.
arm_available() {
  local arm="$1"
  case "$arm" in
    v3_788|v3_3000) [[ "$V3_AVAILABLE" == "1" ]] ;;
    v4b_788) [[ "$V4B_AVAILABLE" == "1" ]] ;;
    v4b_788_hard) [[ "$V4B_AVAILABLE" == "1" && "$HARD_AVAILABLE" == "1" ]] ;;
    v4b_788_hardsmall) [[ "$V4B_AVAILABLE" == "1" && "$HARD_SMALL_AVAILABLE" == "1" ]] ;;
    v4b_margin_788) [[ "$V4B_MARGIN_AVAILABLE" == "1" ]] ;;
    v4b_all_788) [[ "$V4B_ALL_AVAILABLE" == "1" ]] ;;
    *) return 0 ;;
  esac
}

# ARMS env filter (space/comma list of arm names); empty ARMS = run everything.
arm_selected() {
  local arm="$1"
  if [[ -z "$ARMS" ]]; then return 0; fi
  local a
  local IFS=' ,'
  for a in $ARMS; do
    if [[ "$a" == "$arm" ]]; then return 0; fi
  done
  return 1
}

build_cmd() {
  local arm="$1" gold="$2" n_train="$3" seed="$4"
  local out="$ARMS_DIR/${arm}_${seed}.pt"
  local eval_flags="--eval-gold ${GOLD_V2}"
  case "$arm" in
    v2_788)
      if [[ "$V3_AVAILABLE" == "1" ]]; then eval_flags="$eval_flags --eval-gold-alt ${GOLD_V3}"; fi
      ;;
    v3_788|v3_3000)
      eval_flags="$eval_flags --eval-gold-alt ${GOLD_V3}"
      ;;
    v4b_788)
      if [[ "$V4B_AVAILABLE" == "1" ]]; then eval_flags="$eval_flags --eval-gold-alt ${GOLD_V4B}"; fi
      ;;
    v4b_788_hard|v4b_788_hardsmall)
      # arms-3(b): dev-select (--keep-best) on v4b targets, not v2 -- these
      # arms train ON v4b gold (topped up with hard-gold), so v4b is the
      # natural reference; --eval-gold-alt v2 kept for continuity.
      eval_flags="--eval-gold ${GOLD_V4B} --eval-gold-alt ${GOLD_V2}"
      ;;
    v4b_margin_788|v4b_all_788)
      # gold=v4b_margin/v4b_all (richer-than-top-1 v4b variants); score the
      # holdout against v2 targets (continuity with the v2_788/v3_788
      # reference numbers) AND v4b top-1 targets (equal-gold-quality
      # comparison against the v4b_788 arm). NOTE: --eval-gold also drives
      # the periodic --eval-every dev-holdout target (train_encoder.py
      # dev_eval_gold_path = args.eval_gold or args.gold), so with
      # --eval-gold set to v2 here, keep-best selection is actually done
      # against v2 dev targets, not v4b top-1 -- same as the existing
      # v4b_788 arm's dev selection.
      if [[ "$V4B_AVAILABLE" == "1" ]]; then eval_flags="$eval_flags --eval-gold-alt ${GOLD_V4B}"; fi
      ;;
  esac
  local extra_eval_flag=""
  if [[ "$arm" == "v4b_788_hard" || "$arm" == "v4b_788_hardsmall" ]]; then
    extra_eval_flag="--extra-eval ${HARD_GOLD_TEST_FILLER},${HARD_GOLD_TEST_TEMPLATE}"
  fi
  # arms-3(b): keep ALL of the (small) hard-gold file's records plus 788
  # real v4b records, rather than sampling n_train=788 from the UNION pool
  # (which would under-represent whichever gold file is smaller -- see
  # scripts/train_encoder.py's --n-train-per-file). One count per file in
  # ${gold}'s comma list, same order: 788 real, then ALL of the hard file.
  local n_train_per_file_flag=""
  if [[ "$arm" == "v4b_788_hard" ]]; then
    local hard_n; hard_n="$(wc -l < "$HARD_GOLD_TRAIN")"
    n_train_per_file_flag="--n-train-per-file 788,${hard_n}"
  elif [[ "$arm" == "v4b_788_hardsmall" ]]; then
    local hard_small_n; hard_small_n="$(wc -l < "$HARD_GOLD_TRAIN_SMALL")"
    n_train_per_file_flag="--n-train-per-file 788,${hard_small_n}"
  fi
  local eval_every_flags="--eval-every ${EVAL_EVERY} --dev-holdout-file ${HOLDOUT_DEV_FILE}"
  # lead directive 2026-09-08 (dev/USVS_GRADED_SCORING.md): every arm prints
  # the USVS-graded P/R/F alongside the old binary edge-F1 from now on.
  # LOSS=usvs-soft additionally swaps in the graded soft CE targets.
  local loss_flag=""
  if [[ "$LOSS" != "default" ]]; then loss_flag="--loss ${LOSS} --loss-temperature ${LOSS_TEMPERATURE}"; fi
  local keep_best_flag=""
  if [[ "$KEEP_BEST" == "1" ]]; then keep_best_flag="--keep-best"; fi
  echo "python scripts/train_encoder.py --gold ${gold} --n-train ${n_train}" \
       "--seed ${seed} --subset-seed ${seed} --max-steps ${STEPS} --max-seconds ${MAX_SECONDS}" \
       "--holdout-file ${HOLDOUT_FILE} ${eval_flags} ${eval_every_flags} ${keep_best_flag} ${extra_eval_flag}" \
       "${n_train_per_file_flag} --beam-width ${BEAM_WIDTH} --k ${K} --metric ${METRIC} ${loss_flag} --out ${out}"
}

METRIC="${METRIC:-both}"
LOSS="${LOSS:-default}"
LOSS_TEMPERATURE="${LOSS_TEMPERATURE:-0.25}"

echo "=== encoder training arms ==="
echo "STEPS=$STEPS  MAX_SECONDS=$MAX_SECONDS  PARALLEL=$PARALLEL"
echo "GOLD_V2=$GOLD_V2  GOLD_V3=$GOLD_V3 (available=$V3_AVAILABLE)  GOLD_V4B=$GOLD_V4B (available=$V4B_AVAILABLE)  HARD_GOLD_TRAIN=$HARD_GOLD_TRAIN (available=$HARD_AVAILABLE)"
echo "HOLDOUT_FILE=$HOLDOUT_FILE  HOLDOUT_DEV_FILE=$HOLDOUT_DEV_FILE"
echo "EVAL_EVERY=$EVAL_EVERY  KEEP_BEST=$KEEP_BEST  ARMS=${ARMS:-<all>}"
echo "METRIC=$METRIC  LOSS=$LOSS (T=$LOSS_TEMPERATURE)"
echo

# Build ALL (arm x seed) commands unconditionally -- shown in full even
# when a gold file is missing or an arm was filtered out by ARMS, so
# --dry-run documents the whole plan. Only the RUNNABLE subset (arms with
# all their gold files present AND selected by ARMS) is actually executed
# below.
ALL_COMMANDS=()
RUNNABLE_COMMANDS=()
SKIPPED_ARMS=()
FILTERED_ARMS=()
for arm_def in "${ARM_DEFS[@]}"; do
  IFS=':' read -r arm gold n_train <<< "$arm_def"
  for seed in $SEEDS; do
    cmd="$(build_cmd "$arm" "$gold" "$n_train" "$seed")"
    entry="$arm|$seed|$gold|$n_train|$cmd"
    ALL_COMMANDS+=("$entry")
    if ! arm_selected "$arm"; then
      : # not runnable this invocation (ARMS filter)
    elif ! arm_available "$arm"; then
      : # not runnable (missing gold file)
    else
      RUNNABLE_COMMANDS+=("$entry")
    fi
  done
  if ! arm_available "$arm"; then
    SKIPPED_ARMS+=("$arm")
  elif ! arm_selected "$arm"; then
    FILTERED_ARMS+=("$arm")
  fi
done

if [[ "${#SKIPPED_ARMS[@]}" -gt 0 ]]; then
  echo "SKIP (missing gold file): ${SKIPPED_ARMS[*]} -- commands are still listed below for reference; they will not run."
fi
if [[ "${#FILTERED_ARMS[@]}" -gt 0 ]]; then
  echo "SKIP (not in \$ARMS): ${FILTERED_ARMS[*]}"
fi

echo "Planned commands (${#ALL_COMMANDS[@]} total, ${#RUNNABLE_COMMANDS[@]} runnable now):"
for entry in "${ALL_COMMANDS[@]}"; do
  IFS='|' read -r arm seed gold n_train cmd <<< "$entry"
  tag=""
  if ! arm_available "$arm"; then tag=" [SKIP: missing gold]"; fi
  if ! arm_selected "$arm"; then tag="$tag [SKIP: not in \$ARMS]"; fi
  echo "  [$arm seed=$seed]$tag $cmd"
done
echo

COMMANDS=("${RUNNABLE_COMMANDS[@]}")
if [[ "${#COMMANDS[@]}" -eq 0 ]]; then
  echo "No arms to run."
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
import json, sys, torch
arm, seed, gold, n_train, out, summary = sys.argv[1:7]
ckpt = torch.load(out, map_location="cpu", weights_only=False)
cfg = ckpt["config"]
m = ckpt["metrics"].get("holdout", {})
alt = ckpt["metrics"].get("holdout_alt", {})

def fmt(v):
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return ""

filler_json, template_json = "", ""
for gp, fam_metrics in (ckpt.get("extra_eval") or {}).items():
    compact = {fam: (round(fm["rank1_edge_f1"], 4) if fm["rank1_edge_f1"] == fm["rank1_edge_f1"] else None)
               for fam, fm in fam_metrics.items()}
    if "filler" in gp:
        filler_json = json.dumps(compact)
    elif "template" in gp:
        template_json = json.dumps(compact)

row = [
    arm, seed, gold, n_train,
    str(cfg.get("optimizer_steps")), str(cfg.get("stop_reason")),
    f"{cfg.get('epoch_fraction', float('nan')):.4f}",
    f"{ckpt.get('train_wallclock_s', float('nan')):.1f}",
    fmt(m.get("edge_precision")), fmt(m.get("edge_recall")),
    fmt(m.get("rank1_edge_f1")), fmt(m.get("mean_forest_width")),
    fmt(alt.get("edge_precision")), fmt(alt.get("rank1_edge_f1")),
    str(cfg.get("best_step")), fmt(cfg.get("best_dev_rank1_f1")), fmt(cfg.get("last_dev_rank1_f1")),
    fmt(cfg.get("epochs_equivalent")),
    filler_json, template_json,
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
