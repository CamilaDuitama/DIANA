#!/bin/bash
# Step 2: Aggregate k-mer counts to unitig-level features using kmat_tools

set -e

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <kmer_counts> <unitigs_fa> <kmer_size> <out_abundance> [out_fraction]"
    echo ""
    echo "Aggregate k-mer counts to unitig level using existing kmat_tools unitig"
    exit 1
fi

KMER_COUNTS=$1
UNITIGS_FA=$2
KMER_SIZE=$3
OUT_ABUNDANCE=$4
OUT_FRACTION=$5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_NAME=$(basename "$OUT_ABUNDANCE" | sed 's/_unitig_abundance.txt//')

# Resolve kmat_tools. The project env comes FIRST: an unrelated kmat_tools on the
# user's PATH (e.g. a personal miniconda) takes different arguments and fails with
# "unrecognized arguments: -p ... --out-frac" only after the expensive k-mer
# counting step has already run. CLAUDE.md: there is one environment, use it.
# KMAT_TOOLS_BIN, when set, names the binary explicitly: features must be aggregated by
# the same kmat_tools version that built the training matrix (v19 C5: the matrix was
# built by MUSET v0.6.1; env/bin/kmat_tools is v0.5.1).
PROJECT_KMAT="$SCRIPT_DIR/../../env/bin/kmat_tools"
SUBMODULE_KMAT="$SCRIPT_DIR/../../external/muset/bin/kmat_tools"
if [ -n "${KMAT_TOOLS_BIN:-}" ]; then
    [ -x "$KMAT_TOOLS_BIN" ] || { echo "[ERROR] KMAT_TOOLS_BIN=$KMAT_TOOLS_BIN is not executable"; exit 2; }
    KMAT_TOOLS="$KMAT_TOOLS_BIN"
elif [ -x "$PROJECT_KMAT" ]; then
    KMAT_TOOLS="$PROJECT_KMAT"
elif [ -x "$SUBMODULE_KMAT" ]; then
    KMAT_TOOLS="$SUBMODULE_KMAT"
elif command -v kmat_tools >/dev/null 2>&1; then
    KMAT_TOOLS="$(command -v kmat_tools)"
    echo "[WARN] Using kmat_tools from PATH ($KMAT_TOOLS); ./env/bin/kmat_tools not found."
else
    echo "[ERROR] kmat_tools not found at $PROJECT_KMAT, $SUBMODULE_KMAT, or on PATH"
    exit 2
fi

echo "[INFO] kmat_tools: $KMAT_TOOLS ($("$KMAT_TOOLS" --version 2>&1 | head -1))"

# Fail fast if the resolved binary is not the MUSET one that supports --out-frac.
if ! "$KMAT_TOOLS" unitig --help 2>&1 | grep -q -- '--out-frac'; then
    echo "[ERROR] $KMAT_TOOLS does not support 'unitig --out-frac'."
    echo "        This is not the MUSET kmat_tools. Expected ./env/bin/kmat_tools (v0.5.x)."
    exit 2
fi

OUTPUT_PREFIX="${OUT_ABUNDANCE%_abundance.txt}"

# Build command as an array to avoid eval and quoting hazards
KMAT_CMD=(
    "$KMAT_TOOLS" unitig
    -k "$KMER_SIZE"
    -p "$OUTPUT_PREFIX"
)

if [ -n "$OUT_FRACTION" ]; then
    KMAT_CMD+=(--out-frac)
fi

KMAT_CMD+=("$UNITIGS_FA" "$KMER_COUNTS")

"${KMAT_CMD[@]}"

# Rename outputs to match expected names
if [ -f "${OUTPUT_PREFIX}.abundance.mat" ]; then
    # Extract just the single column (skip unitig ID column)
    awk '{print $2}' "${OUTPUT_PREFIX}.abundance.mat" > "$OUT_ABUNDANCE"
    rm "${OUTPUT_PREFIX}.abundance.mat"
fi

if [ -n "$OUT_FRACTION" ] && [ -f "${OUTPUT_PREFIX}.frac.mat" ]; then
    awk '{print $2}' "${OUTPUT_PREFIX}.frac.mat" > "$OUT_FRACTION"
    rm "${OUTPUT_PREFIX}.frac.mat"
fi
