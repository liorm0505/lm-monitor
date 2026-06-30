#!/bin/bash
# LM Monitor - Start All (Dashboard + Log Server)
# Double-click this file to start everything

echo "🚀 Starting LM Monitor..."
echo "   Dashboard: http://localhost:8080"
echo "   Log Server: http://localhost:8081"
echo ""

# Find the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment if it exists
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "✅ Virtual environment activated"
fi

# Run the dashboard (this also starts the log server)
cd "$SCRIPT_DIR"
python3 llm_monitor.py

echo ""
echo "⚠️  LM Monitor stopped"
echo "   Press Enter to close..."
read
