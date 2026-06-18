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
| **Total Latency** | Time from request to complete response — what you actually experience |
| **Generation Speed** | Tokens/sec — how fast the model generates output |

## How We Measure

We probe LM Studio with a short non-streaming request (`"stream": false`). This is important because:

- **LM Studio prioritizes non-streaming requests** over streaming ones
- With concurrency=1, a streaming test ping would get queued behind your actual chat requests → inflated numbers
- The response includes `usage` data (prompt_tokens, completion_tokens) and we measure elapsed time via `response.elapsed`
- Total latency = prompt processing + generation combined (the real-world number that matters)

## What Each Metric Tells You

- **Total Latency:** How long you wait from sending a message to getting the full response. Lower is better — dominated by model size, GPU offload, and RAM pressure.
- **Generation Speed:** Tokens per second during output. On Apple Silicon this is typically memory-bandwidth bound — faster with smaller models or less GPU offload.

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
