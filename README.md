# LM Monitor Dashboard

Lightweight Python HTTP dashboard for monitoring macOS Memory Pressure, RAM usage, and LM Studio inference speed (tokens/sec + total latency) remotely via iPhone.

## Quick Install on Mac Mini M4 Pro

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

## Metrics Shown

| Metric | Description |
|--------|-------------|
| **Memory Pressure** | macOS memory pressure state (Low/Medium/High) |
| **RAM Usage** | Current RAM usage percentage + GB available/total |
| **Avg Generation Speed** | Tokens/sec averaged over last 10 real requests from LM Studio logs |
| **Avg Prompt Processing** | Milliseconds to process the prompt, averaged over last 10 requests |

## How We Measure

The dashboard reads **actual LM Studio server logs** (not test pings) to get real-world metrics:

- Logs are stored at `~/.lmstudio/server-logs/YYYY-MM/YYYY-MM-DD.N.log`
- Each chat completion request produces timing lines like:
  ```
  prompt eval time = 58880.48 ms / 19027 tokens (323.15 tokens per second)
  eval time =    3486.10 ms /   113 tokens (32.41 tokens per second)
  ```
- The script parses these lines, groups metrics by task ID, and calculates averages over the last 10 requests

### Required LM Studio Settings

You **must** enable **"Verbose Server Logs"** in LM Studio:
1. Open LM Studio → **Settings** → **Developer**
2. Toggle **"Verbose Logging"** ON
3. Set **"File Logging Mode"** to **"Succinct"** (or any mode)
4. Restart the LM Studio server

Without this, the log files won't contain the timing data needed for metrics.

## Configuration

Edit `llm_monitor.py` to change:
- `PORT` — Dashboard port (default: 8080)
- `LM_STUDIO_URL` — LM Studio server URL (default: `http://localhost:1234`)
- `CACHE_TTL` — Cache duration in seconds (default: 5)

## Troubleshooting

- **Dashboard not loading?** Check firewall allows port 8080.
- **LM Studio speed shows "N/A"?** Ensure LM Studio is running with remote connections enabled.
- **Memory pressure shows "Unknown"?** Script requires macOS — won't work on Linux/Windows.

## Author

Hermes Agent · Nous Research
