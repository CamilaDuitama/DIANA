#!/usr/bin/env bash
# test_integration.sh — End-to-end smoke test for a fresh DIANA install.
#
# Runs diana-predict on the bundled small test sample (ERR3609654) and
# asserts that all expected output files are created and well-formed.
#
# Usage:
#   bash test_integration.sh            # run from repo root
#   bash test_integration.sh --verbose  # show diana-predict stdout/stderr
#
# Prerequisites (fulfilled by install.sh):
#   - diana-predict in PATH  (via ./env created by install.sh)
#   - results/training/best_model.pth      (downloaded from Hugging Face)
#   - models/pca_reference.pkl             (downloaded from Hugging Face)
#   - training_matrix/unitigs.fa           (downloaded from Zenodo)
#   - training_matrix/reference_kmers.fasta (downloaded from Zenodo)

set -eo pipefail

# Resolve repo root (tests/ is one level below the root)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$REPO_ROOT"
VERBOSE=false
[[ "${1:-}" == "--verbose" ]] && VERBOSE=true

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
pass()  { echo -e "${GREEN}[PASS]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*" >&2; FAILURES=$((FAILURES+1)); }
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[SKIP]${NC}  $*"; }

FAILURES=0

# ── prerequisite checks ───────────────────────────────────────────────────────
info "Checking prerequisites…"

MATRIX="$SCRIPT_DIR/training_matrix"
KMERS="$MATRIX/reference_kmers.fasta"
UNITIGS="$MATRIX/unitigs.fa"

# Locate best_model.pth: prefer this repo, fall back to sibling DIANA repo
MODEL="$SCRIPT_DIR/results/training/best_model.pth"
if [[ ! -f "$MODEL" ]]; then
    for candidate in \
        "$(dirname "$REPO_ROOT")/DIANA/results/training/best_model.pth" \
        "$(dirname "$(dirname "$REPO_ROOT")")/DIANA/results/training/best_model.pth"
    do
        if [[ -f "$candidate" ]]; then
            MODEL="$candidate"
            info "Using model from: $MODEL"
            break
        fi
    done
fi

missing=0
for f in "$MODEL" "$KMERS" "$UNITIGS"; do
    if [[ ! -f "$f" ]]; then
        warn "Missing prerequisite: $f"
        if [[ "$f" == "$UNITIGS" ]]; then
            warn "  unitigs.fa is downloaded by install.sh from Zenodo."
            warn "  Make sure you have run: bash install.sh"
        fi
        missing=$((missing+1))
    fi
done

# Resolve diana-predict: prefer PATH, then ./env, then a sibling DIANA/env
# (created by: mamba env create -f environment.yml -p ./env && bash install.sh)
DIANA_PREDICT=""
if command -v diana-predict &>/dev/null; then
    DIANA_PREDICT="diana-predict"
elif [[ -x "$REPO_ROOT/env/bin/diana-predict" ]]; then
    DIANA_PREDICT="$REPO_ROOT/env/bin/diana-predict"
    info "Using local env: $DIANA_PREDICT"
else
    # Look for a DIANA env in sibling directories (e.g. ../../DIANA/env)
    for candidate in \
        "$(dirname "$REPO_ROOT")/DIANA/env/bin/diana-predict" \
        "$(dirname "$(dirname "$REPO_ROOT")")/DIANA/env/bin/diana-predict" \
        "$HOME/scratch/DIANA/env/bin/diana-predict"
    do
        if [[ -x "$candidate" ]]; then
            DIANA_PREDICT="$candidate"
            info "Using env: $DIANA_PREDICT"
            break
        fi
    done
fi

if [[ -z "$DIANA_PREDICT" ]]; then
    warn "diana-predict not found in PATH, $REPO_ROOT/env/bin/, or any sibling DIANA/env/"
    warn "Create the environment first:"
    warn "  mamba env create -f environment.yml -p ./env"
    warn "  mamba activate ./env"
    warn "  bash install.sh"
    missing=$((missing+1))
fi

if [[ $missing -gt 0 ]]; then
    echo ""
    echo "Run install.sh first, then re-run this script."
    exit 2
fi

# Prepend the env's bin dir to PATH so all subprocesses (kmat_tools,
# back_to_sequences, python, etc.) come from the right environment.
DIANA_ENV_BIN="$(dirname "$DIANA_PREDICT")"
export PATH="$DIANA_ENV_BIN:$PATH"
info "Using env bin: $DIANA_ENV_BIN"

pass "All prerequisites present"

# ── pick test sample ──────────────────────────────────────────────────────────
TESTDATA="$SCRIPT_DIR/test_data"
SAMPLE_ID="ERR3609654"

# Prefer small files if present, fall back to full files
R1=$(ls "$TESTDATA/${SAMPLE_ID}_1_small.fastq.gz" \
         "$TESTDATA/${SAMPLE_ID}_1.fastq.gz" 2>/dev/null | head -1 || true)
R2=$(ls "$TESTDATA/${SAMPLE_ID}_2_small.fastq.gz" \
         "$TESTDATA/${SAMPLE_ID}_2.fastq.gz" 2>/dev/null | head -1 || true)

if [[ -z "$R1" || -z "$R2" ]]; then
    fail "Test FASTQ files not found in $TESTDATA/"
    echo "Expected: ${SAMPLE_ID}_1[_small].fastq.gz and ${SAMPLE_ID}_2[_small].fastq.gz"
    exit 1
fi

info "Test sample : $SAMPLE_ID"
info "R1          : $(basename "$R1")"
info "R2          : $(basename "$R2")"

# ── run diana-predict ─────────────────────────────────────────────────────────
OUTDIR="$SCRIPT_DIR/test_results/integration_${SAMPLE_ID}"
rm -rf "$OUTDIR"

info "Running diana-predict…"
CMD=(
    $DIANA_PREDICT
    --sample "$R1" "$R2"
    --model  "$MODEL"
    --training-matrix "$MATRIX"
    --output "$OUTDIR"
    --threads 4
    --no-plots
)

if $VERBOSE; then
    "${CMD[@]}"
else
    "${CMD[@]}" &>/dev/null
fi

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    fail "diana-predict exited with code $EXIT_CODE"
    echo "Re-run with --verbose for details."
    exit 1
fi
pass "diana-predict completed (exit 0)"

# ── assert output structure ───────────────────────────────────────────────────
info "Checking output files…"

SDIR="$OUTDIR/$SAMPLE_ID"

check_file() {
    local f="$1" label="$2"
    if [[ -f "$f" ]]; then
        pass "$label"
    else
        fail "Missing: $label  ($f)"
    fi
}

check_nonempty() {
    local f="$1" label="$2"
    if [[ -f "$f" && -s "$f" ]]; then
        pass "$label (non-empty)"
    elif [[ -f "$f" ]]; then
        fail "Empty file: $label  ($f)"
    else
        fail "Missing: $label  ($f)"
    fi
}

check_nonempty "$SDIR/${SAMPLE_ID}_kmer_counts.txt"      "kmer_counts"
check_nonempty "$SDIR/${SAMPLE_ID}_unitig_abundance.txt"  "unitig_abundance"
check_nonempty "$SDIR/${SAMPLE_ID}_unitig_fraction.txt"   "unitig_fraction"
check_file     "$SDIR/${SAMPLE_ID}_predictions.json"       "predictions.json"

# ── validate JSON structure ───────────────────────────────────────────────────
info "Validating predictions JSON…"
JSON="$SDIR/${SAMPLE_ID}_predictions.json"

python3 - "$JSON" <<'PYEOF'
import json, sys

path = sys.argv[1]
with open(path) as fh:
    data = json.load(fh)

errors = []

# Top-level keys
for key in ("sample_id", "predictions"):
    if key not in data:
        errors.append(f"missing top-level key: '{key}'")

# All four task heads must be present
expected_tasks = {"sample_type", "material", "sample_host", "community_type"}
preds = data.get("predictions", {})
missing_tasks = expected_tasks - set(preds.keys())
if missing_tasks:
    errors.append(f"missing prediction tasks: {missing_tasks}")

# Each task must have predicted_class and confidence
for task, v in preds.items():
    if "predicted_class" not in v:
        errors.append(f"{task}: missing 'predicted_class'")
    conf = v.get("confidence")
    if conf is None:
        errors.append(f"{task}: missing 'confidence'")
    elif not (0.0 <= conf <= 1.0):
        errors.append(f"{task}: confidence {conf} out of [0,1]")

if errors:
    for e in errors:
        print(f"[FAIL]  JSON: {e}", file=sys.stderr)
    sys.exit(1)

# Print a short summary
st = preds.get("sample_type", {})
print(f"[PASS]  sample_type = {st.get('predicted_class')}  "
      f"(confidence {st.get('confidence', 0):.3f})")
for task in ("material", "sample_host", "community_type"):
    t = preds.get(task, {})
    print(f"[PASS]  {task:<18} = {t.get('predicted_class')}  "
          f"(confidence {t.get('confidence', 0):.3f})")
PYEOF

PYEXIT=$?
if [[ $PYEXIT -ne 0 ]]; then
    FAILURES=$((FAILURES+1))
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
if [[ $FAILURES -eq 0 ]]; then
    pass "All checks passed — integration test OK"
    exit 0
else
    fail "$FAILURES check(s) failed"
    exit 1
fi
