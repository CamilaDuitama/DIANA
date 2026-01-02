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
