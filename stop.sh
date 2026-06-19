#!/usr/bin/env bash
# Kill dashboard (port 8080) and log server (port 8081) cleanly
echo "Stopping llm-monitor servers..."

# Kill by port
kill $(lsof -ti :8080) 2>/dev/null
kill $(lsof -ti :8081) 2>/dev/null

# Also kill by PID file if it exists
if [ -f "logs/log_server.pid" ]; then
    kill "$(cat logs/log_server.pid)" 2>/dev/null
fi

echo "Done. Both servers stopped."
