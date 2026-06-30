#!/bin/bash
# LM Monitor - Start Dashboard
# Double-click this file to start the dashboard and log server

echo "🚀 Starting LM Monitor..."
echo ""

# Find the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment if it exists
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "✅ Virtual environment activated"
fi

# Run the dashboard
cd "$SCRIPT_DIR"
python3 llm_monitor.py

echo ""
echo "⚠️  Dashboard stopped"
echo "Press Enter to close..."
read
