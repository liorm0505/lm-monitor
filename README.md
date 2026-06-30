# LM Monitor Dashboard

Lightweight Python HTTP dashboard for monitoring macOS Memory Pressure, RAM usage, and LM Studio inference speed (tokens/sec + total latency) remotely via iPhone.

## Quick Install on Host Machine (Mac Mini M4 Pro)

### 1. Clone the repo

```bash
cd /home/lior                          # or wherever you want it
git clone https://github.com/liorm0505/lm-monitor.git
cd lm-monitor
```

This creates:
```
/home/lior/lm-monitor/          ← repo root
│
├── llm_monitor.py              ← dashboard script ✅ (in git)
├── requirements.txt            ← psutil + requests ✅ (in git)
├── .gitignore                  ← excludes venv, cache, secrets ✅ (in git)
├── README.md                   ← this file ✅ (in git)
└── .venv/                      ← virtual env ❌ (NOT in git — ignored)
```

### 2. Create the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `.venv/` folder lives alongside the script but is **ignored by git** (see `.gitignore`). It stays local and never gets pushed to GitHub.

### 3. Run it

```bash
python3 llm_monitor.py
```

## Access Dashboard

Open `http://<YOUR_MAC_IP>:8080` from your iPhone Safari.

### Find Your Mac's IP

```bash
ipconfig getifaddr en0   # Wi-Fi
ipconfig getifaddr en1   # Ethernet
```

## Architecture

The dashboard runs as a single Python HTTP server with two components:

1. **Main dashboard** (port 8080) — Serves the HTML dashboard and API endpoints
2. **Log server** (port 8081) — Serves crash/debug logs remotely via HTTP

### How It Works

The dashboard reads **actual LM Studio server logs** (not test pings) to get real-world metrics:

- Logs are stored at `~/.lmstudio/server-logs/YYYY-MM/YYYY-MM-DD.N.log`
- Each chat completion request produces timing lines like:
  ```
  prompt eval time = 58880.48 ms / 19027 tokens (323.15 tokens per second)
  eval time =    3486.10 ms /   113 tokens (32.41 tokens per second)
  ```
- The script parses these lines, groups metrics by task ID, and calculates averages over the last 10 requests

### Key Components

- **`llm_monitor.py`** — Main script with HTTP server, log parser, and dashboard generator
- **`_read_lm_studio_logs()`** — Parses log files for timing metrics
- **`_calculate_averages()`** — Computes running averages from parsed requests
- **`_get_memory_pressure()`** — Queries macOS memory pressure via `memory_pressure` CLI
- **`generate_html()`** — Creates the responsive HTML dashboard
- **`Handler`** — HTTP request handler with multiple endpoints

### Log Server (Port 8081)

The dashboard automatically starts a background HTTP server on port 8081 that serves:
- `debug.log` — Timestamped debug messages from the main server
- `crash.log` — Unhandled exceptions and crashes
- `capture_stats.json` — Network capture statistics (if used)
- `log_server.pid` — PID file for the log server process

Access remotely via: `http://<YOUR_MAC_IP>:8081/debug.log`

This allows debugging without SSH access to the Mac Mini.

## Metrics Shown

| Metric | Description |
|--------|-------------|
| **Memory Pressure** | macOS memory pressure state (Low/Medium/High) |
| **RAM Usage** | Current RAM usage percentage + GB available/total |
| **Avg Generation Speed** | Tokens/sec averaged over last 10 real requests from LM Studio logs |
| **Avg Prompt Processing** | Milliseconds to process the prompt, averaged over last 10 requests |
| **GPU Utilization** | GPU load percentage (Apple Silicon via powermetrics) |
| **GPU Temperature** | GPU die temperature in Celsius |
| **Commit Hash** | Current git commit hash and timestamp |
| **Uptime** | Server uptime since startup |

## Dashboard Features

### Status Bar
- **Commit hash** — Current git commit
- **Commit timestamp** — Full ISO timestamp of the commit
- **Age** — Relative time (e.g., "5m ago", "2h ago")
- **Uptime** — Server uptime
- **Status** — "● running" indicator

### Buttons
- **🔄 Update** — Pull latest from GitHub and restart automatically
- **📋 Model Info** — Show LM Studio model details (no inference needed)
- **🐛 Debug Toggle** — Enable/disable debug logging
- **📤 Forward Logs** — Display log content in browser for debugging
- **↻ Refresh** — Reload the dashboard page

### Auto-Update Mechanism
1. Click update button → `/api/update` endpoint
2. Create backup of current script
3. `git pull origin main`
4. Validate new script compiles
5. Restore backup if validation fails
6. Restart server with new code
7. Log server survives restart (runs in separate process)

### Auto-Rollback
If the script fails validation on startup:
1. Check backup directory for previous versions
2. Restore the most recent backup
3. Restart with the backup version
4. If no backup exists, exit with error

## Configuration

Edit `llm_monitor.py` to change:
- `PORT` — Dashboard port (default: 8080)
- `LM_STUDIO_URL` — LM Studio server URL (default: `http://localhost:1234`)
- `CACHE_TTL` — Cache duration in seconds (default: 5)
- `AVG_WINDOW` — Running average window size (default: 10)

## Required LM Studio Settings

You **must** enable **\"Verbose Server Logs\"** in LM Studio:
1. Open LM Studio → **Settings** → **Developer**
2. Toggle **\"Verbose Logging\"** ON
3. Set **\"File Logging Mode\"** to **\"Succinct\"** (or any mode)
4. Restart the LM Studio server

Without this, the log files won't contain the timing data needed for metrics.

## Debugging

### Check Logs Remotely
```bash
curl http://10.100.102.204:8081/debug.log | tail -50
curl http://10.100.102.204:8081/crash.log
```

### Common Issues

**"local variable 'prompt_tokens' referenced before assignment"**
- This is a Python scoping bug in the log parser
- The variable is defined inside an `if` block but referenced outside it
- Fixed by initializing variables with `.get()` calls at the start of the loop

**"Address already in use" crash**
- Multiple instances of the script running simultaneously
- Kill all instances: `pkill -f llm_monitor.py`
- Then restart: `python3 llm_monitor.py`

**Memory pressure at 5-12%**
- System is under heavy memory pressure
- Check if LM Studio is using too much RAM
- Consider closing other applications
- The dashboard will show "High" memory pressure when this occurs

**Log files not found**
- Ensure LM Studio has "Verbose Server Logs" enabled
- Check that `~/.lmstudio/server-logs/` directory exists
- Verify files are being written to the correct month directory

### Debug Endpoints

- `/debug/toggle?enable=1` — Enable debug logging
- `/debug/toggle?enable=0` — Disable debug logging
- `/debug/logs` — Get last 100 log entries as JSON
- `/api/log_forward` — Get last 100 lines from newest log file

## Troubleshooting

- **Dashboard not loading?** Check firewall allows port 8080.
- **LM Studio speed shows "N/A"?** Ensure LM Studio is running with remote connections enabled.
- **Memory pressure shows "Unknown"?** Script requires macOS — won't work on Linux/Windows.
- **Log server not responding?** Check if port 8081 is listening: `lsof -i :8081`
- **Update button not working?** Verify git credentials and network connectivity

## Author

Hermes Agent · Nous Research
