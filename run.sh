#!/usr/bin/env bash
# ==============================================================================
# Multimodal Disaster Intelligence and Damage Assessment System
# Single-Command Launcher
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=================================================================="
echo "🚀 Starting Multimodal Disaster Intelligence System..."
echo "=================================================================="

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Ensure necessary directories exist
mkdir -p data/uploads data/processed logs models

# Run tests if requested
if [ "$1" == "--test" ]; then
    echo "Running system test suite..."
    pytest tests/ -v
    exit 0
fi

# Launch Streamlit Dashboard
echo "Launching Streamlit Dashboard on http://localhost:8501"
echo "Press Ctrl+C to stop the server."
streamlit run app.py --server.port 8501
