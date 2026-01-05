#!/bin/bash
# DIANA Installation Script
# Compiles external tools and installs them into the active Conda environment

set -e

echo "============================================"
echo "   DIANA Installation"
echo "============================================"

# Check if a conda/mamba environment is active
if [ -z "$CONDA_PREFIX" ]; then
    echo "[ERROR] No conda/mamba environment is active."
    echo "Please activate the DIANA environment first:"
    echo "  mamba activate ./env"
    echo "  OR"
    echo "  conda activate ./env"
    exit 1
fi

if [[ "$CONDA_DEFAULT_ENV" != *"DIANA"* && "$CONDA_DEFAULT_ENV" != *"diana"* && "$CONDA_DEFAULT_ENV" != *"/env"* ]]; then
    echo "[WARNING] Current environment is '$CONDA_DEFAULT_ENV'"
    echo "Make sure you've activated the correct environment (./env)."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Installing to: $CONDA_PREFIX/bin"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Build back_to_sequences (Rust)
# ============================================================================
echo "[1/2] Building back_to_sequences..."
cd "$SCRIPT_DIR/external/back_to_sequences"

if ! command -v cargo >/dev/null 2>&1; then
    echo "[ERROR] cargo not found. Please install Rust or add it to environment.yml"
    exit 1
fi

# Clean previous builds
rm -rf target
cargo clean

# Unset CC to let cargo find the compiler automatically
unset CC
export CC=gcc

# Build (without CPU-specific optimizations for portability)
cargo build --release

# Verify build succeeded
if [ ! -f "target/release/back_to_sequences" ]; then
    echo "[ERROR] Failed to build back_to_sequences"
    exit 1
fi

# Install to conda bin
cp target/release/back_to_sequences "$CONDA_PREFIX/bin/"
chmod +x "$CONDA_PREFIX/bin/back_to_sequences"
echo "✓ Installed back_to_sequences to $CONDA_PREFIX/bin/"
echo ""

# ============================================================================
# Note: MUSET tools are installed via conda package (no build needed)
# ============================================================================
echo "[2/2] Verifying MUSET tools (installed via conda)..."
if ! command -v kmat_tools >/dev/null 2>&1; then
    echo "[WARNING] kmat_tools not found. Make sure muset is installed:"
    echo "  mamba install -c camiladuitama muset"
fi
if ! command -v muset >/dev/null 2>&1; then
    echo "[WARNING] muset not found. Make sure muset is installed:"
    echo "  mamba install -c camiladuitama muset"
fi
echo "✓ MUSET tools available from conda package"
echo ""

# ============================================================================
# Download reference k-mers from Zenodo (if not present)
# ============================================================================
echo "[3/3] Checking reference k-mers file..."

# Define the target location, Zenodo URL, and the expected checksum
KMER_FILE="$SCRIPT_DIR/data/matrices/large_matrix_3070_with_frac/reference_kmers.fasta"
KMER_URL="YOUR_ZENODO_DOWNLOAD_URL_HERE"  # <-- PASTE YOUR ZENODO URL HERE (should end in .fasta.gz)
EXPECTED_CHECKSUM="YOUR_SHA256_CHECKSUM_HERE"  # <-- PASTE SHA256 OF THE .GZ FILE HERE

# Ensure the target directory exists
mkdir -p "$(dirname "$KMER_FILE")"

# Check if the file already exists and is valid
NEEDS_DOWNLOAD=false
if [ -f "$KMER_FILE" ]; then
    echo "Reference k-mers file found. Verifying integrity..."
    ACTUAL_CHECKSUM=$(sha256sum "$KMER_FILE" | awk '{print $1}')
    if [ "$ACTUAL_CHECKSUM" == "$EXPECTED_CHECKSUM" ]; then
        echo "✓ File is valid. Skipping download."
    else
        echo "⚠️  Checksum mismatch. Re-downloading the file."
        rm "$KMER_FILE"  # Remove corrupted file before downloading
        NEEDS_DOWNLOAD=true
    fi
else
    NEEDS_DOWNLOAD=true
fi

if [ "$NEEDS_DOWNLOAD" = true ]; then
    if [ "$KMER_URL" = "YOUR_ZENODO_DOWNLOAD_URL_HERE" ]; then
        echo "[INFO] Reference k-mers file not found: $KMER_FILE"
        echo "[INFO] Zenodo URL not configured yet"
        echo ""
        echo "TO GENERATE AND UPLOAD TO ZENODO:"
        echo "  1. Generate the file:"
        echo "     bash scripts/inference/00_extract_reference_kmers.sh \\"
        echo "          data/matrices/large_matrix_3070_with_frac \\"
        echo "          reference_kmers.fasta"
        echo "  2. Compress: gzip reference_kmers.fasta"
        echo "  3. Calculate checksum: sha256sum reference_kmers.fasta.gz"
        echo "  4. Upload to Zenodo (approx. 179 MB compressed)"
        echo "  5. Update KMER_URL and EXPECTED_CHECKSUM in install.sh"
        echo ""
        echo "[INFO] For now, the file will be generated on first inference run (slower)"
    else
        echo "Downloading reference k-mers (approx. 179 MB compressed)..."
        KMER_GZ="$KMER_FILE.gz"
        
        if command -v wget >/dev/null 2>&1; then
            if ! wget -O "$KMER_GZ" "$KMER_URL"; then
                echo "[ERROR] Download failed. Please check your internet connection and the URL."
                rm -f "$KMER_GZ"  # Clean up partial download
                exit 1
            fi
        elif command -v curl >/dev/null 2>&1; then
            if ! curl -L -o "$KMER_GZ" "$KMER_URL"; then
                echo "[ERROR] Download failed. Please check your internet connection and the URL."
                rm -f "$KMER_GZ"  # Clean up partial download
                exit 1
            fi
        else
            echo "[ERROR] Neither wget nor curl found. Cannot download reference file."
            echo "Please download manually from: $KMER_URL"
            echo "Save to: $KMER_GZ and decompress with: gunzip $KMER_GZ"
            exit 1
        fi
        
        echo "Verifying downloaded file..."
        ACTUAL_CHECKSUM=$(sha256sum "$KMER_GZ" | awk '{print $1}')
        
        if [ "$ACTUAL_CHECKSUM" != "$EXPECTED_CHECKSUM" ]; then
            echo "[ERROR] Checksum mismatch! The downloaded file may be corrupt."
            rm "$KMER_GZ"
            exit 1
        fi
        echo "✓ Download verified."
        
        echo "Decompressing file..."
        gunzip "$KMER_GZ"
        
        if [ ! -f "$KMER_FILE" ]; then
            echo "[ERROR] Decompression failed"
            exit 1
        fi
        
        echo "✓ Reference k-mers ready."
    fi
fi
echo ""

# ============================================================================
# Verify installation
# ============================================================================
echo "============================================"
echo "   Verifying Installation"
echo "============================================"

ALL_GOOD=true

if command -v back_to_sequences >/dev/null 2>&1; then
    echo "✓ back_to_sequences found in PATH"
else
    echo "✗ back_to_sequences NOT found in PATH"
    ALL_GOOD=false
fi

if command -v kmat_tools >/dev/null 2>&1; then
    KMAT_VERSION=$(kmat_tools --version 2>&1 | head -1 || echo "unknown")
    echo "✓ kmat_tools found in PATH ($KMAT_VERSION)"
else
    echo "✗ kmat_tools NOT found in PATH"
    ALL_GOOD=false
fi

if command -v muset >/dev/null 2>&1; then
    MUSET_VERSION=$(muset --version 2>&1 | head -1 || echo "unknown")
    echo "✓ muset found in PATH ($MUSET_VERSION)"
else
    echo "✗ muset NOT found in PATH"
    ALL_GOOD=false
fi

echo "============================================"

if [ "$ALL_GOOD" = true ]; then
    echo ""
    echo "✓ Installation complete!"
    echo ""
    echo "You can now run the DIANA inference pipeline:"
    echo "  scripts/inference/inference_pipeline.sh <config_file>"
    echo ""
else
    echo ""
    echo "✗ Installation incomplete. Please check the errors above."
    exit 1
fi
