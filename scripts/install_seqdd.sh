#!/bin/bash
# Install seqdd for downloading SRA data

set -e

echo "Installing seqdd..."
echo "=================="

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Current Python version: $PYTHON_VERSION"

# Check if Python >= 3.10
PYTHON_MAJOR=$(python -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "ERROR: seqdd requires Python >=3.10, but you have Python $PYTHON_VERSION"
    echo ""
    echo "Please upgrade your conda environment:"
    echo "  conda install python=3.11"
    echo ""
    echo "Or create a new environment:"
    echo "  conda create -n seqdd_env python=3.11"
    echo "  conda activate seqdd_env"
    echo "  ./scripts/install_seqdd.sh"
    exit 1
fi

# Navigate to external directory
cd external/

# Clone seqdd if not already present
if [ ! -d "seqdd" ]; then
    echo "Cloning seqdd repository..."
    git clone https://github.com/yoann-dufresne/seqdd.git
else
    echo "seqdd directory already exists, pulling latest..."
    cd seqdd
    git pull
    cd ..
fi

# Install seqdd into the current environment (not user site-packages)
echo "Installing seqdd into current environment..."
cd seqdd
pip install --no-user .

# Verify installation
echo ""
echo "Verifying installation..."
if seqdd --help > /dev/null 2>&1; then
    echo "✓ seqdd installed successfully!"
    seqdd --help | head -20
else
    echo "✗ seqdd installation failed"
    exit 1
fi
