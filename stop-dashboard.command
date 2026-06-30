#!/bin/bash
# LM Monitor - Stop Dashboard
# Double-click this file to stop the dashboard and log server

echo "🛑 Stopping LM Monitor..."

# Kill dashboard process
if pgrep -f "python3.*llm_monitor.py" > /dev/null; then
    echo "  Killing dashboard..."
    pkill -f "python3.*llm_monitor.py"
fi

# Kill log server
if pgrep -f "python.*-c.*llm_monitor" > /dev/null; then
    echo "  Killing log server..."
    pkill -f "python.*-c.*llm_monitor"
fi

# Kill by port (fallback)
fuser -k 8080/tcp 2>/dev/null
fuser -k 8081/tcp 2>/dev/null

echo "✅ All LM Monitor processes stopped"
echo ""
echo "Press Enter to close..."
read
