#!/bin/bash
# Build MUSET from source with SSHash dictionary saving modification

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MUSET_DIR="$PROJECT_ROOT/external/muset"

echo "=========================================="
echo "Building MUSET with Diana modifications"
echo "=========================================="

# Check if MUSET source exists
if [ ! -d "$MUSET_DIR" ]; then
    echo "ERROR: MUSET source not found at $MUSET_DIR"
    exit 1
fi

cd "$MUSET_DIR"

# Load build dependencies if module system available
if command -v module >/dev/null 2>&1; then
    echo "Loading build dependencies..."
    module load gcc/13.2.0 2>/dev/null || true
    module load cmake/3.31.5 2>/dev/null || true
fi

# Build following README instructions
echo "Building MUSET..."
mkdir -p build && cd build
CC=gcc CXX=g++ cmake ..
make -j4

# Check if build was successful
if [ -f "../bin/muset" ]; then
    echo ""
    echo "✅ MUSET built successfully!"
    echo "Executable: $MUSET_DIR/bin/muset"
    echo ""
    
    # Test with dummy data using production flags
    echo "=========================================="
    echo "Testing with dummy data and production flags..."
    cd ../test
    
    # Activate muset_env for GGCAT (mamba environment)
    eval "$(conda shell.bash hook)"
    mamba activate muset_env
    
    # Make sure we use the same GCC runtime that we built with
    module load gcc/13.2.0 2>/dev/null || true
    export LD_LIBRARY_PATH=/opt/gensoft/exe/gcc/13.2.0/lib64:$LD_LIBRARY_PATH
    
    rm -rf test_output
    
    $MUSET_DIR/bin/muset \
        --file fof.txt \
        -o test_output \
        -k 31 \
        -m 15 \
        -a 2 \
        -l 61 \
        -r 0 \
        --out-frac \
        -t 4 \
        --keep-temp
    
    echo ""
    echo "Test outputs:"
    ls -lh test_output/
    
    # Verify critical files
    echo ""
    echo "Verifying outputs:"
    [ -f test_output/unitigs.abundance.mat ] && echo "  ✓ unitigs.abundance.mat" || echo "  ✗ unitigs.abundance.mat MISSING"
    [ -f test_output/unitigs.frac.mat ] && echo "  ✓ unitigs.frac.mat (NEW!)" || echo "  ✗ unitigs.frac.mat MISSING"
    [ -f test_output/unitigs.sshash.dict ] && echo "  ✓ unitigs.sshash.dict (Diana mod)" || echo "  ✗ unitigs.sshash.dict MISSING"
    
    echo ""
    echo "=========================================="
    echo "✅ Build and test complete!"
    echo "=========================================="
else
    echo "❌ Build failed. muset executable not found."
    exit 1
fi
