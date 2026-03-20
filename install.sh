#!/bin/bash
# DIANA Installation Script
# Builds external tools, downloads the trained model and PCA reference (~382 MB from
# Hugging Face Hub), and downloads reference k-mers (~179 MB from Zenodo) and
# reference unitigs (~18 MB from Zenodo).

set -eo pipefail

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_REPO="cduitamag/DIANA"

MODEL_FILE="$SCRIPT_DIR/results/training/best_model.pth"
MODEL_FILENAME="best_model.pth"
MODEL_CHECKSUM="ef686f1fa07c8d717605fb11a2480eadfa360df64d4ce4419e0ee33e6ec71943"

PCA_FILE="$SCRIPT_DIR/models/pca_reference.pkl"
PCA_FILENAME="pca_reference.pkl"
PCA_CHECKSUM="4bb3f80312b92b113b3f3007820ab3ae59416a531f94129e557fe0ef97f74071"

KMER_FILE="$SCRIPT_DIR/training_matrix/reference_kmers.fasta"
KMER_URL="https://zenodo.org/records/18157419/files/reference_kmers.fasta.gz"
# sha256 of the DECOMPRESSED reference_kmers.fasta (the .gz is removed after install)
KMER_CHECKSUM="9759bb3965466b7e434e72f0726b95ad562e8f7405f2c10d6664e8500b24e1dc"

UNITIGS_FILE="$SCRIPT_DIR/training_matrix/unitigs.fa"
# TODO: replace with the real Zenodo URL once unitigs.fa is uploaded
UNITIGS_URL="https://zenodo.org/records/18157419/files/unitigs.fa"
UNITIGS_CHECKSUM="5d784fc4954643711c6dadece31e4499e5d788766937c37881936bb7cec550b4"

# ============================================================================
# Colored logging helpers
# ============================================================================
RED='\033[0;31m'; YELLOW='\033[0;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }

# ============================================================================
# Helper: download and verify a file (with optional gunzip)
#
# Usage: download_and_verify <label> <url> <dest_file> <expected_sha256> [gunzip]
#   gunzip: if set to "gunzip", the downloaded .gz is decompressed to <dest_file>
#           and the sha256 checksum is verified against the decompressed file.
# ============================================================================
download_and_verify() {
    local label="$1" url="$2" dest="$3" expected_sha256="$4" do_gunzip="${5:-}"

    mkdir -p "$(dirname "$dest")"

    # If already present, verify and skip
    if [ -f "$dest" ]; then
        local actual
        actual=$(sha256sum "$dest" | awk '{print $1}')
        if [ "$actual" = "$expected_sha256" ]; then
            ok "$label already present and verified."
            return 0
        else
            warn "$label checksum mismatch — re-downloading."
            rm -f "$dest"
        fi
    fi

    # Choose download tool
    local gz_dest="$dest"
    [ -n "$do_gunzip" ] && gz_dest="${dest}.gz"

    info "Downloading $label..."
    if command -v wget >/dev/null 2>&1; then
        wget --show-progress -q -O "$gz_dest" "$url" || { error "wget failed for $label."; rm -f "$gz_dest"; exit 1; }
    elif command -v curl >/dev/null 2>&1; then
        curl -L --progress-bar -o "$gz_dest" "$url" || { error "curl failed for $label."; rm -f "$gz_dest"; exit 1; }
    else
        error "Neither wget nor curl found. Install one and retry."
        exit 1
    fi

    # Decompress if requested, then verify the decompressed file
    if [ -n "$do_gunzip" ]; then
        info "Decompressing $label..."
        gunzip "$gz_dest"
        [ -f "$dest" ] || { error "Decompression of $label failed."; exit 1; }
    fi

    # Verify checksum against the final (possibly decompressed) file
    info "Verifying $label..."
    local actual
    actual=$(sha256sum "$dest" | awk '{print $1}')
    if [ "$actual" != "$expected_sha256" ]; then
        error "$label checksum mismatch. The downloaded file may be corrupt."
        rm -f "$dest"
        exit 1
    fi

    ok "$label downloaded and verified."
}

# ============================================================================
# Step 1 — Prerequisites
# ============================================================================
echo ""
echo "============================================"
echo "  DIANA Installation"
echo "============================================"
echo ""
info "Step 1/4 — Checking prerequisites"

if [ -z "$CONDA_PREFIX" ]; then
    error "No conda/mamba environment is active."
    echo "        Please activate the DIANA environment first:"
    echo "          mamba activate ./env   OR   conda activate ./env"
    exit 1
fi

if [[ "$CONDA_DEFAULT_ENV" != *"DIANA"* && "$CONDA_DEFAULT_ENV" != *"diana"* && "$CONDA_DEFAULT_ENV" != *"/env"* ]]; then
    warn "Current environment is '$CONDA_DEFAULT_ENV' (expected the DIANA env at ./env)."
    read -r -p "        Continue anyway? (y/N) " reply
    [[ "$reply" =~ ^[Yy]$ ]] || exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
    error "cargo not found. Rust must be installed (it is declared in environment.yml)."
    echo "        Try: mamba install -c conda-forge rust"
    exit 1
fi

for tool in kmat_tools muset; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        error "$tool not found. Make sure muset is installed in the active environment."
        echo "        Try: mamba install -c camiladuitama muset"
        exit 1
    fi
done

ok "All prerequisites satisfied (conda env: $CONDA_DEFAULT_ENV)"
echo ""

# ============================================================================
# Step 2 — Build back_to_sequences (Rust)
# ============================================================================
info "Step 2/4 — Building back_to_sequences"

BUILD_DIR="$SCRIPT_DIR/external/back_to_sequences"
BUILD_LOG="$BUILD_DIR/build.log"

cd "$BUILD_DIR"
rm -rf target
unset CC; export CC=gcc

info "Running cargo build --release (output → build.log)..."
if ! cargo build --release >"$BUILD_LOG" 2>&1; then
    error "Cargo build failed. Last 30 lines of build.log:"
    tail -30 "$BUILD_LOG" >&2
    exit 1
fi

BINARY="$BUILD_DIR/target/release/back_to_sequences"
[ -f "$BINARY" ] || { error "Binary not found after build."; exit 1; }
install -m 755 "$BINARY" "$CONDA_PREFIX/bin/back_to_sequences"

cd "$SCRIPT_DIR"
ok "back_to_sequences installed to $CONDA_PREFIX/bin/"
echo ""

# ============================================================================
# Step 3 — Download model and PCA reference from Hugging Face Hub
# ============================================================================
info "Step 3/4 — Downloading model and PCA reference from Hugging Face Hub"

download_and_verify \
    "Trained model (~336 MB)" \
    "https://huggingface.co/$HF_REPO/resolve/main/$MODEL_FILENAME" \
    "$MODEL_FILE" \
    "$MODEL_CHECKSUM"

download_and_verify \
    "PCA reference (~46 MB)" \
    "https://huggingface.co/$HF_REPO/resolve/main/$PCA_FILENAME" \
    "$PCA_FILE" \
    "$PCA_CHECKSUM"

echo ""

# ============================================================================
# Step 4 — Download reference k-mers and unitigs from Zenodo
# ============================================================================
info "Step 4/4 — Downloading reference k-mers and unitigs from Zenodo"

download_and_verify \
    "Reference k-mers (~179 MB compressed)" \
    "$KMER_URL" \
    "$KMER_FILE" \
    "$KMER_CHECKSUM" \
    "gunzip"

download_and_verify \
    "Reference unitigs (~18 MB)" \
    "$UNITIGS_URL" \
    "$UNITIGS_FILE" \
    "$UNITIGS_CHECKSUM"

echo ""

# ============================================================================
# Verification summary
# ============================================================================
echo "============================================"
echo "  Verification"
echo "============================================"

ALL_GOOD=true

check_cmd() {
    if command -v "$1" >/dev/null 2>&1; then
        ok "$1 is in PATH"
    else
        error "$1 NOT found in PATH"
        ALL_GOOD=false
    fi
}

check_file() {
    if [ -f "$1" ]; then
        ok "$2 exists"
    else
        error "$2 NOT found at: $1"
        ALL_GOOD=false
    fi
}

check_cmd back_to_sequences
check_cmd kmat_tools
check_cmd muset
check_cmd diana-predict
check_cmd diana-project

check_file "$MODEL_FILE"  "Trained model      (results/training/best_model.pth)"
check_file "$PCA_FILE"    "PCA reference      (models/pca_reference.pkl)"
check_file "$KMER_FILE"     "Reference k-mers   (training_matrix/reference_kmers.fasta)"
check_file "$UNITIGS_FILE"  "Reference unitigs  (training_matrix/unitigs.fa)"

echo ""
if [ "$ALL_GOOD" = true ]; then
    ok "Installation complete! Run the Quick Start in the README to test your setup."
else
    error "Installation incomplete — see errors above."
    exit 1
fi
