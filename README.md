# LM Monitor Dashboard

Lightweight Python HTTP dashboard for monitoring macOS Memory Pressure, RAM usage, and LM Studio inference speed (tokens/sec + TTFT) remotely via iPhone.

## Quick Install on Mac Mini M4 Pro

```bash
# 1. Create directory & copy the script
mkdir -p ~/lm-monitor
cp llm_monitor.py ~/lm-monitor/

# 2. Install dependencies
cd ~/lm-monitor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run it
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
| **LM Studio Speed** | Tokens/sec + TTFT (Time to First Token) via API ping |

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
