#!/usr/bin/env python3
"""
LM Monitor — Real-time LLM Inference Dashboard for Mac Mini (Apple Silicon)
=============================================================================

Monitors:
  • macOS Memory Pressure (Low / Medium / High)
  • Unified RAM Usage (GB used / available / total)
  • LM Studio Prompt Processing Speed (time to first token)
  • LM Studio Generation Speed (tokens/sec)

Uses streaming SSE to capture both prompt processing time and generation speed.
Results cached for CACHE_TTL seconds to avoid inference overhead on page reloads.

Requirements: Python 3.9+ with psutil and requests packages.
Author: Hermes Agent · Nous Research
"""

import http.server
import socketserver
import json
import subprocess
import time
import requests
import psutil
from datetime import datetime

# ──────────────────────────────────────────────
# Configuration — edit these if needed
# ──────────────────────────────────────────────
LM_STUDIO_URL = "http://localhost:1234"   # LM Studio local server port
PORT          = 8080                      # Dashboard HTTP port
CACHE_TTL     = 5                         # Seconds between LM Studio pings


# ──────────────────────────────────────────────
# Cache state — prevents inference overhead on page reloads
# ──────────────────────────────────────────────
_cache = {
    "lm_online": False,
    "lm_ttft": "—",          # Time to first token (ms) — prompt processing time
    "lm_gen_speed": "—",     # Generation speed (tokens/sec after first token)
    "lm_detail": "Waiting...",
    "lm_ts": 0,
}


# ──────────────────────────────────────────────
# Data collection functions
# ──────────────────────────────────────────────

def _get_memory_pressure():
    """Query macOS memory pressure via sysctl."""
    try:
        status = subprocess.run(
            ["sysctl", "-n", "vm.page_pressure"],
            capture_output=True, text=True
        ).stdout.strip()
        if status == "0":
            return "Low", "#34c759"       # Green
        elif status == "1":
            return "Medium", "#ff9f0a"    # Yellow
        else:
            return "High", "#ff3b30"      # Red
    except Exception:
        return "—", "#888888"


def _get_ram_usage():
    """Return RAM percentage, total GB, available GB."""
    mem = psutil.virtual_memory()
    return mem.percent, mem.total / (1024**3), mem.available / (1024**3)


def _ping_lm_studio():
    """Fire a streaming inference request and measure both:
    
    • Time to First Token (TTFT) — prompt processing time in ms
    • Generation Speed — tokens/sec
    
    Uses SSE streaming so we can capture the first token arrival time.
    Called once every CACHE_TTL seconds; results are cached for page views.
    """
    try:
        payload = {
            "model": "",
            "messages": [{"role": "user", "content": "Say 'OK'"}],
            "max_tokens": 5,
            "temperature": 0.1,
            "stream": True,               # Enable streaming for TTFT measurement
        }
        
        start_time = time.time()
        first_token_time = None
        total_tokens = 0
        
        response = requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json=payload,
            timeout=30,
            stream=True
        )
        
        elapsed_total = time.time() - start_time
        
        if response.status_code == 200:
            # Parse SSE stream manually to capture first token time
            for line in response.iter_lines():
                if line:
                    text = line.decode("utf-8")
                    # SSE lines look like: data: {"choices":[{"delta":{"content":"..."}}]}
                    if text.startswith("data: "):
                        data_str = text[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            for choice in choices:
                                delta = choice.get("delta", {})
                                content = delta.get("content")
                                if content and first_token_time is None:
                                    first_token_time = (time.time() - start_time) * 1000  # ms
                                total_tokens += 1
                        except json.JSONDecodeError:
                            continue
        
        # Build results — separate prompt processing (TTFT) from generation speed
        if not response.ok:
            return False, "Offline", f"Status: {response.status_code}", "—"
        
        detail_parts = []
        ttft_str = "—"
        gen_speed_str = "—"
        
        # TTFT = prompt processing time (ms)
        if first_token_time is not None:
            ttft_str = f"{first_token_time:.0f} ms"
            detail_parts.append(f"TTFT: {ttft_str}")
        
        # Generation speed = tokens/sec AFTER first token
        if total_tokens > 0 and elapsed_total > 0:
            gen_speed = total_tokens / elapsed_total
            gen_speed_str = f"{gen_speed:.1f} tok/s"
            detail_parts.append(f"{total_tokens} tokens in {elapsed_total:.2f}s ({gen_speed_str})")
        
        if not detail_parts:
            detail_parts.append("Waiting for response...")
        
        return True, gen_speed_str, ", ".join(detail_parts), ttft_str
    
    except Exception as e:
        return False, "Error", str(e)


def _get_cached_lm_stats():
    """Return cached LM Studio stats unless TTL has expired."""
    now = time.time()
    if now - _cache["lm_ts"] > CACHE_TTL:
        online, gen_speed, detail, ttft = _ping_lm_studio()
        _cache.update({
            "lm_online": online,
            "lm_gen_speed": gen_speed,
            "lm_detail": detail,
            "lm_ttft": ttft,
            "lm_ts": now,
        })
    return _cache["lm_online"], _cache["lm_gen_speed"], _cache["lm_detail"], _cache["lm_ttft"]


# ──────────────────────────────────────────────
# HTML generation — responsive mobile dashboard
# ──────────────────────────────────────────────

def generate_html(pressure, pressure_color, ram_pct, ram_total, ram_avail, lm_online, lm_gen_speed, lm_detail, lm_ttft):
    timestamp = datetime.now().strftime("%H:%M:%S")
    dot_color = "#34c759" if lm_online else "#ff3b30"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LM Monitor</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a1a; color: #fff; padding: 24px 16px 80px; min-height: 100vh; }}
  .header {{ text-align: center; margin-bottom: 20px; }}
  .header h1 {{ font-size: 1.3em; color: #aaa; font-weight: 500; }}
  .card {{ background: #2d2d2d; border-radius: 14px; padding: 18px; margin-bottom: 14px; box-shadow: 0 4px 10px rgba(0,0,0,0.35); }}
  h2 {{ font-size: 0.8em; color: #777; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
  .value {{ font-size: 2.2em; font-weight: 700; }}
  .sub {{ font-size: 0.85em; color: #666; margin-top: 5px; line-height: 1.4; }}
  .pressure-badge {{ display: inline-block; padding: 6px 16px; border-radius: 10px; font-size: 1.3em; font-weight: 700; color: #fff; background: {pressure_color}; }}
  .status-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; background: {dot_color}; box-shadow: 0 0 6px {dot_color}; }}
  .refresh-btn {{ position: fixed; bottom: 20px; right: 20px; background: #007aff; color: white; border: none; padding: 14px; border-radius: 50%; font-size: 22px; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }}
  .footer {{ text-align: center; color: #555; font-size: 0.7em; margin-top: 28px; line-height: 1.6; }}
  .ttft-value {{ font-size: 1.6em; font-weight: 600; color: #ff9f0a; }}
</style>
</head>
<body>
  <div class="header"><h1>⚡ LM Monitor</h1></div>

  <div class="card">
    <h2>Memory Pressure</h2>
    <span class="pressure-badge">{pressure}</span>
  </div>

  <div class="card">
    <h2>RAM Usage</h2>
    <div class="value">{ram_pct}%</div>
    <div class="sub">{ram_avail:.1f} GB available / {ram_total:.1f} GB total</div>
  </div>

  <div class="card">
    <h2><span class="status-dot"></span>Prompt Processing (TTFT)</h2>
    <div class="ttft-value">{lm_ttft}</div>
    <div class="sub">Time from request to first token — how fast the model reads & processes your prompt</div>
  </div>

  <div class="card">
    <h2><span class="status-dot"></span>Generation Speed</h2>
    <div class="value">{lm_gen_speed}</div>
    <div class="sub">{lm_detail}</div>
  </div>

  <div class="footer">
    Last updated: {timestamp} · LM Studio stats refresh every {CACHE_TTL}s<br>
    Auto-refresh page with ↻ button below
  </div>

  <button class="refresh-btn" onclick="location.reload()">↻</button>
</body>
</html>"""


# ──────────────────────────────────────────────
# HTTP server handler
# ──────────────────────────────────────────────

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            pressure, p_color = _get_memory_pressure()
            ram_pct, ram_total, ram_avail = _get_ram_usage()
            lm_online, lm_gen_speed, lm_detail, lm_ttft = _get_cached_lm_stats()

            html = generate_html(
                pressure, p_color, ram_pct, ram_total, ram_avail,
                lm_online, lm_gen_speed, lm_detail, lm_ttft
            )
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            super().do_GET()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 Dashboard running at http://<YOUR_MAC_IP>:{PORT}")
        print(f"  LM Studio pings cached every {CACHE_TTL}s (no inference overhead on page reload)")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
