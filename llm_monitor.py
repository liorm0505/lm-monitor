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
import os
import sys
from datetime import datetime
from io import StringIO

# ──────────────────────────────────────────────
# Startup timestamp — used for uptime tracking
# ──────────────────────────────────────────────
_START_TIME = time.time()


# ──────────────────────────────────────────────
# Git version info — resolves commit hash & timestamp
# ──────────────────────────────────────────────

def _get_git_info():
    """Return (short_hash, commit_timestamp) from git log, or ('—', '—') if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h %ci"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(_SCRIPT_PATH) or "."
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            return parts[0], parts[1]  # short_hash, ISO timestamp
    except Exception:
        pass
    return "—", "—"


def _uptime_str():
    """Return human-readable uptime string."""
    elapsed = int(time.time() - _START_TIME)
    if elapsed < 60:
        return f"{elapsed}s ago"
    elif elapsed < 3600:
        return f"{elapsed // 60}m ago"
    else:
        return f"{elapsed // 3600}h {(elapsed % 3600) // 60}m ago"


# ──────────────────────────────────────────────
# Debug log capture — lightweight stdout/stderr wrapper
# ──────────────────────────────────────────────

class _LogCapture(StringIO):
    """Captures writes to a circular buffer. Only active when enabled."""
    def __init__(self, max_size=200):
        super().__init__()
        self._buffer = []
        self._max = max_size

    def write(self, text):
        if _cache.get("logs_enabled"):
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self._buffer.append({"ts": ts, "msg": text.rstrip("\n")})
            # FIFO: keep only last N entries
            if len(self._buffer) > self._max:
                del self._buffer[:len(self._buffer) - self._max]
        return super().write(text)

    def flush(self):
        pass  # no-op for StringIO

    def get_logs(self):
        return list(self._buffer)


_debug_stdout = _LogCapture()
_debug_stderr = _LogCapture()
_debug_original_stdout = sys.stdout
_debug_original_stderr = sys.stderr


def _toggle_logs(on: bool):
    """Redirect stdout/stderr to capture buffers, or restore originals."""
    global _debug_stdout, _debug_stderr, _debug_original_stdout, _debug_original_stderr
    if on:
        sys.stdout = _debug_stdout
        sys.stderr = _debug_stderr
    else:
        sys.stdout = _debug_original_stdout
        sys.stderr = _debug_original_stderr


# ──────────────────────────────────────────────
# Configuration — edit these if needed
# ──────────────────────────────────────────────
LM_STUDIO_URL = "http://localhost:1234"   # LM Studio local server port
PORT          = 8080                      # Dashboard HTTP port
CACHE_TTL     = 5                         # Seconds between LM Studio pings

# ──────────────────────────────────────────────
# Auto-reload: track our own script mtime at startup
# ──────────────────────────────────────────────
_SCRIPT_PATH = os.path.abspath(__file__)
_SCRIPT_MTIME_START = os.path.getmtime(_SCRIPT_PATH)


# ──────────────────────────────────────────────
# Cache state — prevents inference overhead on page reloads
# ──────────────────────────────────────────────
_cache = {
    "lm_online": False,
    "lm_ttft": "—",          # Time to first token (ms) — prompt processing time
    "lm_gen_speed": "—",     # Generation speed (tokens/sec after first token)
    "lm_detail": "Waiting...",
    "lm_ts": 0,
    "logs_enabled": False,   # Toggle: capture stdout/stderr to /debug/logs
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
                                content = delta.get("content") or delta.get("text")
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
        
        # Generation speed = tokens/sec during generation phase ONLY (after first token)
        if total_tokens > 0 and elapsed_total > 0:
            if first_token_time is not None:
                gen_duration = elapsed_total - (first_token_time / 1000.0)
                if gen_duration > 0:
                    gen_speed = total_tokens / gen_duration
                    gen_speed_str = f"{gen_speed:.1f} tok/s"
                    detail_parts.append(f"{total_tokens} tokens in {gen_duration:.2f}s ({gen_speed_str})")
                else:
                    gen_speed_str = f"{total_tokens/elapsed_total:.1f} tok/s"
                    detail_parts.append(f"{total_tokens} tokens in ~{elapsed_total:.3f}s ({gen_speed_str})")
            else:
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
    commit_hash, commit_ts = _get_git_info()
    uptime = _uptime_str()
    logs_enabled = _cache.get("logs_enabled", False)
    dbg_color = "#ff453a" if logs_enabled else "#8e8e93"  # Red when on, gray when off

    # Format commit time as relative ("2 min ago", "3 days ago", etc.)
    try:
        from datetime import timezone
        commit_dt = datetime.fromisoformat(commit_ts)
        age_seconds = int(time.time() - commit_dt.timestamp())
        if age_seconds < 60:
            commit_age = f"{age_seconds}s ago"
        elif age_seconds < 3600:
            commit_age = f"{age_seconds // 60}m ago"
        elif age_seconds < 86400:
            commit_age = f"{age_seconds // 3600}h ago"
        else:
            commit_age = f"{age_seconds // 86400}d ago"
    except Exception:
        commit_age = "—"

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
  .status-bar {{ display: flex; justify-content: center; align-items: center; gap: 12px; padding: 8px 12px; margin-top: 20px; border-radius: 10px; background: #1e1e1e; border: 1px solid #333; }}
  .status-bar span {{ font-size: 0.75em; color: #888; }}
  .status-bar .commit {{ color: #64d2ff; font-family: 'SF Mono', 'Fira Code', monospace; font-weight: 600; }}
  .status-bar .uptime {{ color: #34c759; }}
  .debug-toggle {{ position: fixed; bottom: 80px; right: 20px; background: {dbg_color}; color: white; border: none; padding: 10px; border-radius: 50%; font-size: 16px; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.5); opacity: 0.7; }}
  .debug-toggle:hover {{ opacity: 1; }}
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

  <div class="status-bar">
    <span class="commit">{commit_hash}</span>
    <span>·</span>
    <span>{commit_age} ago</span>
    <span>·</span>
    <span class="uptime">{uptime}</span>
    <span>·</span>
    <span>● running</span>
  </div>

  <button class="debug-toggle" id="debugBtn" title="Toggle debug logging" onclick="toggleDebug()">🐛</button>

  <script>
    // Toggle debug logging on/off
    function toggleDebug() {{
      const btn = document.getElementById('debugBtn');
      const isOn = btn.style.background === 'rgb(255, 69, 58)';
      fetch('/debug/toggle?enable=' + (isOn ? '0' : '1'))
        .then(() => {{ location.reload(); }})
        .catch(() => {{}});
    }}
  </script>

  <button class="refresh-btn" onclick="location.reload()">↻</button>
</body>
</html>"""


# ──────────────────────────────────────────────
# HTTP server handler
# ──────────────────────────────────────────────

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Auto-reload: if the script file changed since we started, restart ourselves
        global _SCRIPT_MTIME_START
        current_mtime = os.path.getmtime(_SCRIPT_PATH)
        if current_mtime > _SCRIPT_MTIME_START:
            print("🔄 Script changed — reloading…")
            os.execv(sys.executable, [sys.executable, __file__])

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

        elif self.path == "/status":
            uptime = int(time.time() - _START_TIME)
            status_data = {
                "status": "running",
                "uptime_seconds": uptime,
                "pid": os.getpid(),
                "timestamp": datetime.now().isoformat(),
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status_data).encode())

        elif self.path.startswith("/debug/toggle"):
            # Parse query string: /debug/toggle?enable=1 or /debug/toggle?enable=0
            params = self.path.split("?")[1] if "?" in self.path else ""
            enable = "1" in params or "true" in params.lower()
            _cache["logs_enabled"] = enable
            _toggle_logs(enable)
            response_text = "Debug logging enabled" if enable else "Debug logging disabled"
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(response_text.encode())

        elif self.path == "/debug/logs":
            # Return last 100 log entries as JSON
            logs = _debug_stdout.get_logs() + _debug_stderr.get_logs()
            # Sort by timestamp and keep last 100
            logs.sort(key=lambda x: x["ts"])
            logs = logs[-100:]
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "enabled": _cache.get("logs_enabled", False),
                "count": len(logs),
                "logs": logs,
            }, indent=2).encode())

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
