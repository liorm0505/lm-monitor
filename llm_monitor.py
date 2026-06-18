#!/usr/bin/env python3
"""
LM Monitor — Real-time LLM Inference Dashboard for Mac Mini (Apple Silicon)
=============================================================================

Monitors:
  • macOS Memory Pressure (Low / Medium / High)
  • Unified RAM Usage (GB used / available / total)
  • LM Studio Avg Generation Speed (tokens/sec from real requests)
  • LM Studio Avg Prompt Processing Time (from real request logs)

Reads LM Studio's server.log directly from disk — no test pings, no queueing.
Results are running averages over the last N completion requests.

Requirements: Python 3.9+ with psutil and requests packages.
Author: Hermes Agent · Nous Research
"""

import http.server
import socketserver
import json
import subprocess
import time
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
LM_STUDIO_URL = "http://localhost:1234"   # LM Studio local server port (used for /status checks)
PORT          = 8080                      # Dashboard HTTP port
CACHE_TTL     = 5                         # Seconds between log file scans
AVG_WINDOW    = 10                        # Running average over last N completion requests

# ──────────────────────────────────────────────
# Auto-reload: track our own script mtime at startup
# ──────────────────────────────────────────────
_SCRIPT_PATH = os.path.abspath(__file__)
_SCRIPT_MTIME_START = os.path.getmtime(_SCRIPT_PATH)


# ──────────────────────────────────────────────
# LM Studio log path — macOS default location
# ──────────────────────────────────────────────
LM_STUDIO_LOG_PATH = os.path.expanduser(
    "~/Library/Application Support/lm-studio/logs/server.log"
)


# ──────────────────────────────────────────────
# Cache state — prevents frequent file reads on page reloads
# ──────────────────────────────────────────────
_cache = {
    "lm_online": False,
    "lm_gen_speed": "—",       # Avg generation speed (tokens/sec) from real requests
    "lm_detail": "Waiting...", # Detail string with avg metrics
    "lm_ttft": "—",            # Avg prompt processing time (ms) — placeholder for now
    "lm_ts": 0,
    "logs_enabled": False,     # Toggle: capture stdout/stderr to /debug/logs
    "log_file_exists": None,   # Whether we found the log file on startup
    "recent_requests": [],     # Last N parsed completion requests (in memory)
}


# ──────────────────────────────────────────────
# Data collection — read LM Studio server.log
# ──────────────────────────────────────────────

def _read_lm_studio_logs():
    """Read and parse the last lines of LM Studio's server.log.
    
    Returns a list of dicts, each representing a parsed /v1/chat/completions response.
    Each dict contains: timestamp, prompt_tokens, completion_tokens, duration_ms, status
    """
    results = []
    
    # Check if log file exists (cached on first run)
    if _cache["log_file_exists"] is False:
        print("📋 LM Studio log not found — skipping scan. "
              "Enable 'Verbose Server Logs' in LM Studio settings.")
        return results
    
    try:
        with open(LM_STUDIO_LOG_PATH, "r", errors="replace") as f:
            lines = f.readlines()
        
        print(f"📋 Read {len(lines)} lines from server.log")
        
        # Process lines in reverse (newest first) to find completions quickly
        for line in reversed(lines[-500:]):  # Last 500 lines
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Filter for chat completion responses
            path = entry.get("path", "") or entry.get("url", "")
            if "/v1/chat/completions" not in path:
                continue
            
            status = entry.get("status", 0)
            if status != 200:
                continue
            
            # Extract metrics from response
            response_data = entry.get("response", {})
            usage = response_data.get("usage", {})
            
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            if not completion_tokens:
                continue
            
            # Duration from the log entry
            duration_ms = entry.get("duration_ms", 0) or entry.get("response_time_ms", 0)
            
            # Timestamp from the log entry
            ts = entry.get("timestamp", "") or entry.get("time", "")
            
            request_data = entry.get("request", {})
            model = request_data.get("model", "")
            
            req_entry = {
                "timestamp": ts,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get("total_tokens", prompt_tokens + completion_tokens),
                "duration_ms": duration_ms,
                "model": model,
                "status": status,
            }
            
            results.append(req_entry)
            print(f"  ✅ Parsed: {completion_tokens} tokens, {duration_ms}ms, model={model}")
    
    except FileNotFoundError:
        _cache["log_file_exists"] = False
        print("⚠️  LM Studio log file not found at:", LM_STUDIO_LOG_PATH)
        print("   Enable 'Verbose Server Logs' in LM Studio → Settings → Developer")
    except PermissionError:
        print(f"❌ Permission denied reading: {LM_STUDIO_LOG_PATH}")
        _cache["log_file_exists"] = False
    except Exception as e:
        print(f"❌ Error reading log: {e}")
    
    return results


def _calculate_averages(requests):
    """Calculate running averages from parsed completion requests.
    
    Returns (avg_gen_speed, avg_duration_ms, detail_string, sample_count)
    """
    if not requests:
        return "—", "—", "No completion requests found in logs", 0
    
    total_tokens = sum(r["completion_tokens"] for r in requests)
    total_time = sum(r["duration_ms"] for r in requests)
    
    if total_time <= 0:
        gen_speed = "—"
    else:
        gen_speed = f"{total_tokens / (total_time / 1000):.1f} tok/s"
    
    avg_duration = total_time / len(requests)
    
    # Average prompt tokens
    avg_prompt = sum(r["prompt_tokens"] for r in requests) / len(requests)
    avg_completion = sum(r["completion_tokens"] for r in requests) / len(requests)
    
    detail_parts = [
        f"{len(requests)} recent requests",
        f"Avg: {avg_prompt:.0f} prompt tokens → {avg_completion:.0f} completion tokens",
        f"Total: {total_tokens} tokens in {total_time/1000:.1f}s ({gen_speed if isinstance(gen_speed, str) else '—'})",
    ]
    
    return gen_speed, avg_duration, ", ".join(detail_parts), len(requests)


def _get_cached_lm_stats():
    """Return cached LM Studio stats unless TTL has expired."""
    now = time.time()
    if now - _cache["lm_ts"] > CACHE_TTL:
        # Read and parse log file
        requests = _read_lm_studio_logs()
        
        # Update in-memory recent requests (keep last 50)
        _cache["recent_requests"] = requests[-50:]
        
        # Calculate averages over the last AVG_WINDOW requests
        window = _cache["recent_requests"][-AVG_WINDOW:] if len(_cache["recent_requests"]) >= AVG_WINDOW else _cache["recent_requests"]
        
        gen_speed, avg_duration_ms, detail, count = _calculate_averages(window)
        
        # Determine online status based on whether we found recent requests
        online = count > 0
        
        _cache.update({
            "lm_online": online,
            "lm_gen_speed": gen_speed,
            "lm_detail": detail,
            "lm_ttft": f"{avg_duration_ms:.0f} ms" if avg_duration_ms != "—" else "—",
            "lm_ts": now,
        })
        
        print(f"📊 Stats updated: {count} requests in window, gen_speed={gen_speed}")
    
    return _cache["lm_online"], _cache["lm_gen_speed"], _cache["lm_detail"], _cache["lm_ttft"]


# ──────────────────────────────────────────────
# System metrics
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
    <h2><span class="status-dot"></span>Avg Generation Speed</h2>
    <div class="value">{lm_gen_speed}</div>
    <div class="sub">{lm_detail}</div>
  </div>

  <div class="card">
    <h2><span class="status-dot"></span>Avg Prompt Processing Time</h2>
    <div class="ttft-value">{lm_ttft}</div>
    <div class="sub">Average duration of last {AVG_WINDOW} completion requests (includes prompt + generation)</div>
  </div>

  <div class="footer">
    Last updated: {timestamp} · LM Studio log scan every {CACHE_TTL}s<br>
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
                "lm_stats": {
                    "online": _cache["lm_online"],
                    "gen_speed": _cache["lm_gen_speed"],
                    "avg_ttft_ms": _cache["lm_ttft"],
                    "detail": _cache["lm_detail"],
                    "recent_requests_count": len(_cache.get("recent_requests", [])),
                },
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status_data, indent=2).encode())

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
            
            # Also include cache state for debugging
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "enabled": _cache.get("logs_enabled", False),
                "count": len(logs),
                "recent_requests": len(_cache.get("recent_requests", [])),
                "log_file_exists": _cache.get("log_file_exists"),
                "lm_studio_log_path": LM_STUDIO_LOG_PATH,
                "logs": logs,
            }, indent=2).encode())

        else:
            super().do_GET()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import psutil
    
    print(f"🚀 Dashboard running at http://<YOUR_MAC_IP>:{PORT}")
    print(f"   Reading LM Studio logs from: {LM_STUDIO_LOG_PATH}")
    print(f"   Log scan every {CACHE_TTL}s · Running avg over last {AVG_WINDOW} requests")
    print("Press Ctrl+C to stop.")
    
    # Check if log file exists on startup
    if os.path.exists(LM_STUDIO_LOG_PATH):
        _cache["log_file_exists"] = True
        print(f"✅ Found LM Studio log: {LM_STUDIO_LOG_PATH}")
    else:
        _cache["log_file_exists"] = False
        print(f"⚠️  LM Studio log not found at: {LM_STUDIO_LOG_PATH}")
        print("   Enable 'Verbose Server Logs' in LM Studio → Settings → Developer")
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass
