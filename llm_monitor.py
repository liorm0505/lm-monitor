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
import traceback
from io import StringIO

# Guarded psutil import — available in module scope for all functions
try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[name-defined]

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
# LM Studio log path — macOS v0.4+ directory structure
# Logs are organized by month: ~/.lmstudio/server-logs/YYYY-MM/
# ──────────────────────────────────────────────
LM_STUDIO_LOG_DIR = os.path.expanduser("~/.lmstudio/server-logs")


def _find_log_files():
    """Find all log files in the current month's directory."""
    import datetime
    
    now = datetime.datetime.now()
    month_dir = os.path.join(LM_STUDIO_LOG_DIR, now.strftime("%Y-%m"))
    
    if not os.path.isdir(month_dir):
        return []
    
    # Return all .log files sorted by name (newest first)
    files = [f for f in os.listdir(month_dir) if f.endswith('.log')]
    files.sort(reverse=True)  # Newest first (YYYY-MM-DD.N.log)
    return [os.path.join(month_dir, f) for f in files]


def _log_file_exists():
    """Check if any log files exist."""
    return len(_find_log_files()) > 0


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
# Data collection — parse LM Studio text-based logs
# ──────────────────────────────────────────────

def _read_lm_studio_logs():
    """Read and parse LM Studio's server logs (text format).
    
    Extracts metrics from 'slot print_timing' lines:
    - prompt eval time = XXXX ms / YYYY tokens (ZZZ.ZZ t/s)
    - eval time = XXXX ms / YY tokens (BB.BB t/s)
    - total time = XXXX ms / ZZZZ tokens
    
    Returns list of dicts with: task_id, prompt_tokens, completion_tokens, 
                                 prompt_time_ms, eval_time_ms, gen_speed_tps
    """
    import re
    
    results = []
    
    # Early exit if no log files found
    if _cache["log_file_exists"] is False:
        print("📋 No LM Studio log files found — skipping scan. "
              "Enable 'Verbose Server Logs' in LM Studio → Settings → Developer")
        return results
    
    # Find all log files for current month
    log_files = _find_log_files()
    if not log_files:
        print("📋 No log files in current month directory — skipping scan.")
        return results
    
    print(f"📋 Found {len(log_files)} log file(s) to scan")
    
    # Regex patterns for extracting metrics
    prompt_re = re.compile(
        r'prompt\s+eval\s+time\s+=\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens.*?(\d+\.\d+)\s+tokens?\s+per\s+second'
    )
    eval_re = re.compile(
        r'eval\s+time\s+=\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens.*?(\d+\.\d+)\s+tokens?\s+per\s+second'
    )
    task_re = re.compile(r'task\s+(\d+)')
    
    # Process each log file (newest first)
    total_lines = 0
    for log_file in log_files[:3]:  # Scan last 3 files max
        try:
            with open(log_file, "r", errors="replace") as f:
                lines = f.readlines()
            total_lines += len(lines)
            
            # Track metrics by task ID (group prompt eval + eval time together)
            task_metrics = {}
            
            for line in lines[-1000:]:  # Last 1000 lines per file
                line = line.strip()
                if not line or 'slot print_timing' not in line:
                    continue
                
                # Extract task ID
                task_match = task_re.search(line)
                if not task_match:
                    continue
                task_id = task_match.group(1)
                
                # Check for prompt eval time
                prompt_match = prompt_re.search(line)
                if prompt_match:
                    prompt_time_ms = float(prompt_match.group(1))
                    prompt_tokens = int(prompt_match.group(2))
                    prompt_tps = float(prompt_match.group(3))
                    
                    if task_id not in task_metrics:
                        task_metrics[task_id] = {}
                    task_metrics[task_id]['prompt_time_ms'] = prompt_time_ms
                    task_metrics[task_id]['prompt_tokens'] = prompt_tokens
                    task_metrics[task_id]['prompt_tps'] = prompt_tps
                
                # Check for eval time (generation)
                eval_match = eval_re.search(line)
                if eval_match:
                    eval_time_ms = float(eval_match.group(1))
                    completion_tokens = int(eval_match.group(2))
                    eval_tps = float(eval_match.group(3))
                    
                    if task_id not in task_metrics:
                        task_metrics[task_id] = {}
                    task_metrics[task_id]['eval_time_ms'] = eval_time_ms
                    task_metrics[task_id]['completion_tokens'] = completion_tokens
                    task_metrics[task_id]['eval_tps'] = eval_tps
            
            # Extract complete results from grouped metrics
            for task_id, metrics in task_metrics.items():
                if 'prompt_tokens' in metrics and 'completion_tokens' in metrics:
                    prompt_tokens = metrics['prompt_tokens']
                    completion_tokens = metrics['completion_tokens']
                    prompt_time_ms = metrics.get('prompt_time_ms', 0)
                    eval_time_ms = metrics.get('eval_time_ms', 0)
                    
                    # Calculate generation speed (tokens/sec during generation phase)
                    if eval_time_ms > 0:
                        gen_speed = completion_tokens / (eval_time_ms / 1000.0)
                    else:
                        gen_speed = 0
                    
                    results.append({
                        'task_id': task_id,
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'prompt_time_ms': prompt_time_ms,
                        'prompt_tps': metrics.get('prompt_tps', 0),   # tokens/sec during prompt processing
                        'eval_time_ms': eval_time_ms,
                        'gen_speed_tps': gen_speed,
                    })

                if task_id in task_metrics and 'prompt_tps' in task_metrics[task_id] and task_metrics[task_id]['prompt_tps'] > 0:
                    print(f"🔍 Task {task_id}: prompt={prompt_tokens} tok ({metrics['prompt_tps']:.0f} t/s) · gen={completion_tokens} tok ({gen_speed:.1f} t/s)")
                    
        except Exception as e:
            print(f"⚠️  Error reading {log_file}: {e}")
    
    print(f"📋 Parsed {len(results)} completion requests from {total_lines} lines")
    return results


def _calculate_averages(requests):
    """Calculate running averages from parsed completion requests.
    
    Returns (avg_gen_speed, avg_prompt_time_ms, detail_string, sample_count)
    """
    if not requests:
        return "—", "—", "No completion requests found in logs", 0
    
    # Use gen_speed_tps directly from logs (already calculated per-request)
    speeds = [r["gen_speed_tps"] for r in requests if r["gen_speed_tps"] > 0]
    prompt_times = [r["prompt_time_ms"] for r in requests if r["prompt_time_ms"] > 0]
    prompt_tps_vals = [r["prompt_tps"] for r in requests if r["prompt_tps"] > 0]

    avg_gen_speed = f"{sum(speeds) / len(speeds):.1f} tok/s" if speeds else "—"
    avg_prompt_time = sum(prompt_times) / len(prompt_times) if prompt_times else 0
    avg_prompt_tps = f"{sum(prompt_tps_vals) / len(prompt_tps_vals):.0f} tok/s" if prompt_tps_vals else "—"

    # Average token counts
    avg_prompt = sum(r["prompt_tokens"] for r in requests) / len(requests)
    avg_completion = sum(r["completion_tokens"] for r in requests) / len(requests)

    detail_parts = [
        f"{len(requests)} recent requests",
        f"Context: {avg_prompt:.0f} tokens · Gen speed: {avg_gen_speed}",
        f"Prompt processing: {avg_prompt_time:.0f} ms avg ({avg_prompt_tps})",
    ]

    return avg_gen_speed, avg_prompt_time, avg_prompt_tps, avg_prompt, ", ".join(detail_parts), len(requests)


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
        
        gen_speed, avg_prompt_time, prompt_tps, avg_context, detail, count = _calculate_averages(window)

        # Determine online status based on whether we found recent requests
        online = count > 0

        _cache.update({
            "lm_online": online,
            "lm_gen_speed": gen_speed,
            "lm_detail": detail,
            "lm_ttft": f"{avg_prompt_time:.0f} ms" if avg_prompt_time != "—" else "—",
            "lm_prompt_tps": prompt_tps,
            "lm_context_size": avg_context,
            "lm_ts": now,
        })

        print(f"📊 Stats updated: {count} requests in window, gen_speed={gen_speed}, prompt_speed={prompt_tps}, context={avg_context:.0f} tokens")
    return _cache["lm_online"], _cache["lm_gen_speed"], _cache["lm_detail"], _cache["lm_ttft"], _cache["lm_prompt_tps"], _cache["lm_context_size"]


# ──────────────────────────────────────────────
# System metrics
# ──────────────────────────────────────────────

def _get_memory_pressure():
    """Query macOS memory pressure via sysctl."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "vm.page_pressure"],
            capture_output=True, text=True
        )
        raw_stdout = repr(result.stdout)
        stderr_val = result.stderr.strip() if result.stderr else "(empty)"
        log_debug(f"MEM_PRESSURE: sysctl exit_code={result.returncode}, stdout={raw_stdout}, stderr={stderr_val}")

        status = result.stdout.strip()
        if not status:
            log_debug("MEM_PRESSURE: ⚠️ empty string — defaulting to '—'")
            return "—", "#888888"

        # Try parsing as integer for Apple Silicon (which may return 0/1/2+)
        try:
            val = int(status)
            log_debug(f"MEM_PRESSURE: parsed int={val}")
        except ValueError:
            log_debug(f"MEM_PRESSURE: ⚠️ not an integer, value='{status}'")

        if status == "0":
            return "Low", "#34c759"       # Green
        elif status == "1":
            return "Medium", "#ff9f0a"    # Yellow
        else:
            log_debug(f"MEM_PRESSURE: falling through to High (value='{status}')")
            return "High", "#ff3b30"      # Red
    except Exception as e:
        log_debug(f"MEM_PRESSURE: EXCEPTION — {type(e).__name__}: {e}")
        return "—", "#888888"


def _get_ram_usage():
    """Return RAM percentage, total GB, available GB."""
    if psutil is None:
        return "—", "—", "—"
    mem = psutil.virtual_memory()
    return mem.percent, mem.total / (1024**3), mem.available / (1024**3)


# ──────────────────────────────────────────────
# HTML generation — responsive mobile dashboard
# ──────────────────────────────────────────────

def generate_html(pressure, pressure_color, ram_pct, ram_total, ram_avail, lm_online, lm_gen_speed, lm_detail, lm_ttft, lm_prompt_tps, lm_context):
    timestamp = datetime.now().strftime("%H:%M:%S")
    dot_color = "#34c759" if lm_online else "#ff3b30"
    commit_hash, commit_ts = _get_git_info()
    uptime = _uptime_str()
    logs_enabled = _cache.get("logs_enabled", False)
    print(f"🐛 DEBUG RENDER: logs_enabled={logs_enabled} (raw cache={_cache.get('logs_enabled', 'MISSING')!r})")
    dbg_color = "#ff453a" if logs_enabled else "#8e8e93"  # Red when on, gray when off
    _state = "on" if logs_enabled else "off"

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

  <div class="card">
    <h2><span class="status-dot"></span>Avg Prompt Processing Speed</h2>
    <div class="value">{lm_prompt_tps}</div>
    <div class="sub">Tokens/sec during prompt evaluation phase</div>
  </div>

  <div class="card">
    <h2><span class="status-dot"></span>Avg Context Size</h2>
    <div class="value">{lm_context:.0f} tokens</div>
    <div class="sub">Average prompt token count over last {AVG_WINDOW} requests</div>
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

  <button class="debug-toggle" id="debugBtn" title="Toggle debug logging" onclick="toggleDebug()" data-state="{_state}">🐛</button>

  <script>
    // Toggle debug logging on/off
    function toggleDebug() {{
      const btn = document.getElementById('debugBtn');
      const isOn = btn.dataset.state === 'on';
      fetch('/debug/toggle?enable=' + (isOn ? '0' : '1'))
        .then(() => {{ location.reload(); }})
        .catch(() => {{}});
    }}

    // Reset aggregation window
    function resetAvg() {{
      if (!confirm('Reset the running average of last {AVG_WINDOW} requests? This will clear cached stats and re-scan from disk.')) return;
      fetch('/reset_avg')
        .then(() => {{ location.reload(); }})
        .catch(() => {{ alert('Failed to reset'); }});
    }}
  </script>

  <div style="text-align:center; margin-top:12px;">
    <button onclick="resetAvg()" style="background:#3a3a3c;color:#fff;border:none;padding:8px 16px;border-radius:8px;font-size:0.9em;cursor:pointer;">Reset Aggregation ({AVG_WINDOW})</button>
  </div>

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
            lm_online, lm_gen_speed, lm_detail, lm_ttft, lm_prompt_tps, lm_context = _get_cached_lm_stats()

            html = generate_html(
                pressure, p_color, ram_pct, ram_total, ram_avail,
                lm_online, lm_gen_speed, lm_detail, lm_ttft, lm_prompt_tps, lm_context
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
                    "prompt_speed": _cache.get("lm_prompt_tps", "—"),
                    "context_size": f"{_cache.get('lm_context_size', 0):.0f} tokens",
                    "detail": _cache["lm_detail"],
                    "recent_requests_count": len(_cache.get("recent_requests", [])),
                },
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status_data, indent=2).encode())

        elif self.path == "/reset_avg":
            # Reset aggregation window: clear recent requests and force cache refetch
            _cache["recent_requests"] = []
            _cache["lm_ts"] = 0
            print("🔄 Aggregation reset — cleared recent_requests, forcing fresh log scan on next request")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "reset"}).encode())

        elif self.path.startswith("/debug/toggle"):
            # Parse query string: /debug/toggle?enable=1 or /debug/toggle?enable=0
            params = self.path.split("?")[1] if "?" in self.path else ""
            enable = "1" in params or "true" in params.lower()
            old_val = _cache.get("logs_enabled", False)
            _cache["logs_enabled"] = enable
            _toggle_logs(enable)
            print(f"🐛 DEBUG TOGGLE: {old_val} → {enable} (params={params!r})")
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
                "log_dir_exists": _cache.get("log_dir_exists"),
                "lm_studio_log_dir": LM_STUDIO_LOG_DIR,
                "logs": logs,
            }, indent=2).encode())

        else:
            super().do_GET()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# Crash handler & debug logging
# ──────────────────────────────────────────────

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

def log_debug(msg: str) -> None:
    """Append a timestamped message to logs/debug.log"""
    os.makedirs(LOGS_DIR, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(os.path.join(LOGS_DIR, "debug.log"), "a") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

def handle_crash(exc_type, exc_value, exc_traceback) -> None:
    """Write unhandled exceptions to logs/crash.log"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(os.path.join(LOGS_DIR, "crash.log"), "a") as f:
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"--- Crashed at {ts} ---\n")
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            f.writelines(tb_lines)
            f.write("\n")
    except Exception:
        pass  # If we can't write, still try below
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_crash

# ──────────────────────────────────────────────
# Background log HTTP server (survives dashboard crashes)
# ──────────────────────────────────────────────

def _start_log_server() -> None:
    """Start a background http.server on port 8081 serving the logs/ folder."""
    import subprocess
    import socket
    
    # Check if already running on port 8081
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", 8081))
    sock.close()
    if result == 0:
        log_debug("Log server already running on port 8081")
        return
    
    # Not running — start it detached from the dashboard process
    pid_file = os.path.join(LOGS_DIR, "log_server.pid")
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8081"],
        cwd=LOGS_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Save PID so stop.sh can kill it cleanly
    with open(pid_file, "w") as f:
        f.write(str(proc.pid))
    log_debug(f"Log server started on port 8081 (PID {proc.pid})")

# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print(f"🚀 Dashboard running at http://<YOUR_MAC_IP>:{PORT}")
    print(f"   Reading LM Studio logs from: {LM_STUDIO_LOG_DIR}/")
    print(f"   Log scan every {CACHE_TTL}s · Running avg over last {AVG_WINDOW} requests")
    print("Press Ctrl+C to stop.")
    
    # Check if log directory exists on startup
    if os.path.isdir(LM_STUDIO_LOG_DIR):
        _cache["log_dir_exists"] = True
        print(f"✅ Found LM Studio log dir: {LM_STUDIO_LOG_DIR}/")
    else:
        _cache["log_dir_exists"] = False
        print(f"⚠️  LM Studio log dir not found at: {LM_STUDIO_LOG_DIR}/")
        print("   Enable 'Verbose Server Logs' in LM Studio → Settings → Developer")
    
    # Start the background log server (survives dashboard crashes)
    _start_log_server()
    print(f"📂 Log server started on port 8081 — I can read crash/debug logs remotely")
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass
