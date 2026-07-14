#!/bin/bash
# Cultural Equilibrium - Dataset Download Script
# ================================================
# Script tự động tải về các datasets cho dự án Cultural Equilibrium
#
# Usage: ./download_datasets.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Cultural Equilibrium - Dataset Download"
echo "========================================"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p culturepark normad raw
echo "✓ Directories created"
echo ""

# Download CulturePark
echo "========================================"
echo "1. Downloading CulturePark Dataset..."
echo "========================================"
if [ -d "culturepark/.git" ]; then
    echo "CulturePark already exists. Pulling latest changes..."
    cd culturepark
    git pull
    cd ..
else
    echo "Cloning CulturePark repository..."
    git clone https://github.com/Scarelette/CulturePark.git culturepark_tmp
    mv culturepark_tmp/* culturepark/
    mv culturepark_tmp/.* culturepark/ 2>/dev/null || true
    rm -rf culturepark_tmp
fi
echo "✓ CulturePark downloaded"
echo "   Location: $SCRIPT_DIR/culturepark"
echo "   Size: $(du -sh culturepark | cut -f1)"
echo ""

# Download NORMAD
echo "========================================"
echo "2. Downloading NORMAD Dataset..."
echo "========================================"
if [ -d "normad/.git" ]; then
    echo "NORMAD already exists. Pulling latest changes..."
    cd normad
    git pull
    cd ..
else
    echo "Cloning NORMAD repository..."
    git clone https://github.com/Akhila-Yerukola/NormAd.git normad_tmp
    mv normad_tmp/* normad/
    mv normad_tmp/.* normad/ 2>/dev/null || true
    rm -rf normad_tmp
fi
echo "✓ NORMAD downloaded"
echo "   Location: $SCRIPT_DIR/normad"
echo "   Size: $(du -sh normad | cut -f1)"
echo ""

# Note about NORMAD encryption
echo "========================================"
echo "⚠️  NORMAD Dataset Note:"
echo "========================================"
echo "The main dataset (datasets.zip.enc) is encrypted."
echo ""
echo "To access the full dataset, you have 3 options:"
echo ""
echo "1. Contact authors for decryption key:"
echo "   - Akhila Yerukola: akhila@seas.upenn.edu"
echo ""
echo "2. Use HuggingFace datasets library:"
echo "   pip install datasets"
echo "   python -c 'from datasets import load_dataset; d = load_dataset(\"akhilayerukola/NormAd\"); d.save_to_disk(\"normad/hf_data\")'"
echo ""
echo "3. Use available samples in:"
echo "   - normad/story_prompts/"
echo "   - normad/src/analysis/"
echo ""

# Summary
echo "========================================"
echo "Download Complete!"
echo "========================================"
echo ""
echo "Dataset Summary:"
echo "  - CulturePark: $(du -sh culturepark | cut -f1)"
echo "  - NORMAD: $(du -sh normad | cut -f1)"
echo "  - Total: $(du -sh . | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. Review data/README.md for usage instructions"
echo "  2. For NORMAD, follow decryption instructions above"
echo "  3. Run preprocessing scripts in code/utils/data_loader.py"
echo ""
echo "========================================"
