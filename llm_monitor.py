#!/usr/bin/env python3
"""
LM Monitor — Real-time LLM Inference Dashboard for Mac Mini (Apple Silicon)
=============================================================================

Monitors:
  • macOS Memory Pressure (Low / Medium / High)
  • Unified RAM Usage (GB used / available / total)
  • LM Studio Avg Generation Speed (tokens/sec from server.log parsing)
  • LM Studio Avg Prompt Processing Time (from server.log parsing)

Parses LM Studio's verbose server.log files for real metrics — no inference probing needed.
Enable 'Verbose Server Logs' in LM Studio → Settings → Developer.

Requirements: Python 3.9+ with psutil and requests packages.
Author: Hermes Agent · Nous Research
"""

import http.server
from http.server import BaseHTTPRequestHandler
import socketserver
import json
import subprocess
import time
import os
import sys
import signal
import threading
from datetime import datetime, timezone
import traceback
from io import StringIO
import urllib.request  # stdlib — always available on Python 3.9+
import urllib.error    # stdlib — always available on Python 3.9+

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
_SCRIPT_PATH = os.path.abspath(__file__)

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
LM_STUDIO_URL   = os.environ.get("LM_STUDIO_URL", "http://localhost:1234")   # LM Studio local server port (API)
PORT            = 8080                      # Dashboard HTTP port
CACHE_TTL       = 5                         # Seconds between API probes
AVG_WINDOW      = 10                        # Running average over last N probe results
BACKUP_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backups")
MAX_BACKUPS     = 5                         # Keep last N backups

# ──────────────────────────────────────────────
# LM Studio log path — macOS v0.4+ directory structure
# Logs are organized by month: ~/.lmstudio/server-logs/YYYY-MM/
# ──────────────────────────────────────────────
LM_STUDIO_LOG_DIR = os.path.expanduser("~/.lmstudio/server-logs")


def _create_backup():
    """Create a timestamped backup of the current script."""
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"llm_monitor_{timestamp}.py")
        
        with open(_SCRIPT_PATH, "r", encoding="utf-8") as src:
            content = src.read()
        with open(backup_path, "w", encoding="utf-8") as dst:
            dst.write(content)
        
        log_debug(f"Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        log_debug(f"Backup creation failed: {type(e).__name__}: {e}")
        return None


def _cleanup_old_backups():
    """Remove oldest backups keeping only MAX_BACKUPS most recent."""
    try:
        if not os.path.exists(BACKUP_DIR):
            return
        
        backups = sorted([
            f for f in os.listdir(BACKUP_DIR) 
            if f.startswith("llm_monitor_") and f.endswith(".py")
        ])
        
        while len(backups) > MAX_BACKUPS:
            oldest = backups.pop(0)
            os.remove(os.path.join(BACKUP_DIR, oldest))
            log_debug(f"Removed old backup: {oldest}")
    except Exception as e:
        log_debug(f"Backup cleanup failed: {type(e).__name__}: {e}")


def _get_latest_backup():
    """Return path to the most recent backup, or None if no backups exist."""
    try:
        if not os.path.exists(BACKUP_DIR):
            return None
        
        backups = sorted([
            f for f in os.listdir(BACKUP_DIR) 
            if f.startswith("llm_monitor_") and f.endswith(".py")
        ])
        
        if not backups:
            return None
        
        return os.path.join(BACKUP_DIR, backups[-1])
    except Exception as e:
        log_debug(f"Backup lookup failed: {type(e).__name__}: {e}")
        return None


def _restore_backup(backup_path):
    """Restore script from a backup file.
    
    Returns True on success, False on failure.
    """
    try:
        if not os.path.exists(backup_path):
            log_debug(f"Backup file not found: {backup_path}")
            return False
        
        with open(backup_path, "r", encoding="utf-8") as src:
            content = src.read()
        with open(_SCRIPT_PATH, "w", encoding="utf-8") as dst:
            dst.write(content)
        
        log_debug(f"Restored from backup: {backup_path}")
        return True
    except Exception as e:
        log_debug(f"Restore failed: {type(e).__name__}: {e}")
        return False


def _validate_script():
    """Check if the current script compiles successfully.
    
    Returns True if valid, False if syntax error or import issue.
    """
    try:
        # Try to compile the script
        with open(_SCRIPT_PATH, "r", encoding="utf-8") as f:
            code = f.read()
        compile(code, _SCRIPT_PATH, "exec")
        
        # Try to import required modules
        import psutil
        import requests
        
        return True
    except SyntaxError as e:
        log_debug(f"Syntax error in script: {e}")
        return False
    except ImportError as e:
        log_debug(f"Missing import: {e}")
        return False
    except Exception as e:
        log_debug(f"Validation error: {type(e).__name__}: {e}")
        return False


# ──────────────────────────────────────────────
# Auto-reload: track our own script mtime at startup
# ──────────────────────────────────────────────
_SCRIPT_MTIME_START = os.path.getmtime(_SCRIPT_PATH)


# ──────────────────────────────────────────────
# LM Studio log path — macOS v0.4+ directory structure
# Logs are organized by month: ~/.lmstudio/server-logs/YYYY-MM/
# ──────────────────────────────────────────────
LM_STUDIO_LOG_DIR = os.path.expanduser("~/.lmstudio/server-logs")


def _find_log_files():
    """Find all log files in the current date's directory (YYYY-MM-DD format)."""
    import datetime
    
    now = datetime.datetime.now()
    
    # Try date-based directory first (YYYY-MM-DD)
    date_dir = os.path.join(LM_STUDIO_LOG_DIR, now.strftime("%Y-%m-%d"))
    if os.path.isdir(date_dir):
        files = [f for f in os.listdir(date_dir) if f.endswith('.log')]
        files.sort(reverse=True)  # Newest first
        return [os.path.join(date_dir, f) for f in files]
    
    # Fallback to month-based directory (YYYY-MM)
    month_dir = os.path.join(LM_STUDIO_LOG_DIR, now.strftime("%Y-%m"))
    if os.path.isdir(month_dir):
        files = [f for f in os.listdir(month_dir) if f.endswith('.log')]
        files.sort(reverse=True)  # Newest first
        return [os.path.join(month_dir, f) for f in files]
    
    # Fallback: search all subdirectories for log files
    if os.path.isdir(LM_STUDIO_LOG_DIR):
        all_files = []
        for subdir in os.listdir(LM_STUDIO_LOG_DIR):
            subdir_path = os.path.join(LM_STUDIO_LOG_DIR, subdir)
            if os.path.isdir(subdir_path):
                for f in os.listdir(subdir_path):
                    if f.endswith('.log'):
                        all_files.append(os.path.join(subdir_path, f))
        all_files.sort(reverse=True)
        return all_files
    
    return []


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
    "lm_ttft": "—",            # Avg prompt processing time (ms)
    "lm_ts": 0,
    "logs_enabled": False,     # Toggle: capture stdout/stderr to /debug/logs
    "log_file_exists": None,   # Whether we found the log file on startup
    "recent_requests": [],     # Last N parsed completion requests (in memory)
}


# ──────────────────────────────────────────────
# Data collection — parse LM Studio text-based logs
# ──────────────────────────────────────────────

def _read_lm_studio_logs(since_ts=None):
    """Read and parse LM Studio's server logs (text format).
    
    Extracts metrics from 'slot print_timing' lines:
    - prompt eval time = XXXX ms / YYYY tokens (ZZZ.ZZ t/s)
    - eval time = XXXX ms / YY tokens (BB.BB t/s)
    - total time = XXXX ms / ZZZZ tokens
    
    Args:
        since_ts: If set, only include entries with timestamps >= this value.
    
    Returns list of dicts with: task_id, prompt_tokens, completion_tokens, 
                                 prompt_time_ms, eval_time_ms, gen_speed_tps
    """
    import re
    
    results = []
    
    ts_re = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    
    def parse_line_ts(line):
        """Extract datetime from log line timestamp, or None."""
        m = ts_re.search(line)
        if not m: return None
        try:
            return datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return None

    if _cache["log_file_exists"] is False:
        print("📋 No LM Studio log files found — skipping scan. "
              "Enable 'Verbose Server Logs' in LM Studio → Settings → Developer")
        return results
    
    log_files = _find_log_files()
    if not log_files:
        print("📋 No log files in current month directory — skipping scan.")
        return results
    
    print(f"📋 Found {len(log_files)} log file(s) to scan")
    
    prompt_re = re.compile(
        r'prompt\s+eval\s+time\s+=\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens.*?(\d+\.\d+)\s+tokens?\s+per\s+second'
    )
    eval_re = re.compile(
        r'eval\s+time\s+=\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens.*?(\d+\.\d+)\s+tokens?\s+per\s+second'
    )
    task_re = re.compile(r'task\s+(\d+)')
    ntokens_re = re.compile(r'n_tokens\s*=\s*(\d+)')
    
    total_lines = 0
    for log_file in log_files[:3]:  # Scan last 3 files max
        try:
            with open(log_file, "r", errors="replace") as f:
                lines = f.readlines()
            total_lines += len(lines)
            
            task_metrics = {}
            
            for line in lines[-1000:]:  # Last 1000 lines per file
                line = line.strip()
                
                if since_ts is not None:
                    line_ts = parse_line_ts(line)
                    if line_ts and line_ts < since_ts:
                        continue
                
                if not line or 'slot print_timing' not in line:
                    continue
                
                task_match = task_re.search(line)
                if not task_match:
                    continue
                task_id = task_match.group(1)
                
                ntokens_match = ntokens_re.search(line)
                if ntokens_match:
                    n_tokens_val = int(ntokens_match.group(1))
                    if task_id not in task_metrics:
                        task_metrics[task_id] = {}
                    task_metrics[task_id]['prompt_tokens'] = n_tokens_val
                
                prompt_match = prompt_re.search(line)
                if prompt_match:
                    prompt_time_ms = float(prompt_match.group(1))
                    prompt_tps = float(prompt_match.group(3))
                    
                    if task_id not in task_metrics:
                        task_metrics[task_id] = {}
                    task_metrics[task_id]['prompt_time_ms'] = prompt_time_ms
                    task_metrics[task_id]['prompt_tps'] = prompt_tps
                
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
            
            for task_id, metrics in task_metrics.items():
                prompt_tokens = metrics.get('prompt_tokens', 0)
                completion_tokens = metrics.get('completion_tokens', 0)
                prompt_time_ms = metrics.get('prompt_time_ms', 0)
                eval_time_ms = metrics.get('eval_time_ms', 0)
                prompt_tps = metrics.get('prompt_tps', 0)
                
                if eval_time_ms > 0:
                    gen_speed = completion_tokens / (eval_time_ms / 1000.0)
                else:
                    gen_speed = 0
                
                if prompt_tokens and completion_tokens:
                    results.append({
                        'task_id': task_id,
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'prompt_time_ms': prompt_time_ms,
                        'prompt_tps': prompt_tps,
                        'eval_time_ms': eval_time_ms,
                        'gen_speed_tps': gen_speed,
                    })
                
                if prompt_tps > 0:
                    print(f"🔍 Task {task_id}: prompt={prompt_tokens} tok ({prompt_tps:.0f} t/s) · gen={completion_tokens} tok ({gen_speed:.1f} t/s)")
                    
        except Exception as e:
            print(f"⚠️  Error reading {log_file}: {e}")
    
    print(f"📋 Parsed {len(results)} completion requests from {total_lines} lines")
    return results


def _calculate_averages(requests):
    """Calculate running averages from parsed completion requests.
    
    Returns (avg_gen_speed, avg_prompt_time_ms, detail_string, sample_count)
    """
    if not requests:
        return "—", "—", "—", "—", "No completion requests found in logs", 0
    
    speeds = [r["gen_speed_tps"] for r in requests if r["gen_speed_tps"] > 0]
    prompt_times = [r["prompt_time_ms"] for r in requests if r["prompt_time_ms"] > 0]
    prompt_tps_vals = [r["prompt_tps"] for r in requests if r["prompt_tps"] > 0]

    avg_gen_speed = f"{sum(speeds) / len(speeds):.1f} tok/s" if speeds else "—"
    avg_prompt_time = sum(prompt_times) / len(prompt_times) if prompt_times else 0
    avg_prompt_tps = f"{sum(prompt_tps_vals) / len(prompt_tps_vals):.0f} tok/s" if prompt_tps_vals else "—"

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
    # During fresh-start (after /reset_avg), skip log scanning until real completions arrive
    if "fresh_start_ts" in _cache:
        reset_time = _cache["fresh_start_ts"]
        now = time.time()
        requests = _read_lm_studio_logs(since_ts=datetime.utcfromtimestamp(reset_time) if isinstance(reset_time, (int, float)) else datetime.now())
        
        if requests:
            del _cache["fresh_start_ts"]
            print("✅ Fresh completions detected post-reset — resuming normal metrics")
        else:
            return False, "—", "No data yet — waiting for completions...", "—", "—", None
    
    now = time.time()
    if now - _cache["lm_ts"] > CACHE_TTL:
        # Read and parse log file
        requests = _read_lm_studio_logs()
        
        _cache["recent_requests"] = requests[-50:]
        
        window = _cache["recent_requests"][-AVG_WINDOW:] if len(_cache["recent_requests"]) >= AVG_WINDOW else _cache["recent_requests"]
        
        gen_speed, avg_prompt_time, prompt_tps, avg_context, detail, count = _calculate_averages(window)

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
        
        print(f"📊 Log scan complete: {count} requests parsed")
    
    return (
        _cache["lm_online"],
        _cache["lm_gen_speed"],
        _cache["lm_detail"],
        _cache["lm_ttft"],
        _cache.get("lm_prompt_tps", "—"),
        _cache.get("lm_context_size", "—"),
    )


# ──────────────────────────────────────────────
# System metrics
# ──────────────────────────────────────────────

def _get_memory_pressure():
    """Query macOS memory pressure via `memory_pressure` CLI (Apple Silicon & Intel)."""
    try:
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode != 0:
            log_debug(f"MEM_PRESSURE: memory_pressure exit={result.returncode}, stderr={repr(result.stderr.strip()[:200])}")
            return _get_memory_pressure_fallback()

        raw = result.stdout.strip()
        log_debug(f"MEM_PRESSURE: raw_stdout={repr(raw)}")

        # macOS memory_pressure CLI can output JSON {\"systemwide_pressure_level\": 0} or plain text "System-wide Memory Pressure: Low"
        level = None

        # 1. Try parsing as JSON first (common on Ventura/Sonoma)
        try:
            import json as _json_mod
            data = _json_mod.loads(raw)
            if isinstance(data, dict):
                level = data.get("systemwide_pressure_level") or data.get("pressure_level")
        except (ValueError, TypeError):
            pass

        # 2. Fallback: grep for text keywords
        if level is None:
            lower_raw = raw.lower()
            if "low" in lower_raw:
                level = 0
            elif "medium" in lower_raw:
                level = 1
            elif "high" in lower_raw:
                level = 2

        # 3. Fallback: parse numeric (some CLI versions just print 0, 1, or 2)
        if level is None:
            try:
                level = int(raw.strip())
            except ValueError:
                pass

        if level == 0:
            return "Low", "#34c759"       # Green — Activity Monitor green bar
        elif level == 1:
            return "Medium", "#ff9f0a"    # Yellow — Activity Monitor yellow bar
        else:
            return "High", "#ff3b30"      # Red — Activity Monitor red bar

    except FileNotFoundError:
        log_debug("MEM_PRESSURE: memory_pressure CLI not found — using psutil estimate")
        return _get_memory_pressure_psutil()
    except Exception as e:
        log_debug(f"MEM_PRESSURE: EXCEPTION — {type(e).__name__}: {e}")
        return _get_memory_pressure_psutil()


def _get_memory_pressure_fallback():
    """Fallback when CLI unavailable: estimate pressure from psutil RAM metrics."""
    if psutil is None:
        return "—", "#888888"
    try:
        mem = psutil.virtual_memory()
        avail_mb = mem.available / (1024 * 1024)

        if avail_mb > 6000:   # > 6 GB still free → Low pressure
            return "Low", "#34c759"
        elif avail_mb > 2000:  # 2–6 GB free → Medium pressure
            return "Medium", "#ff9f0a"
        else:                  # < 2 GB free → High pressure
            return "High", "#ff3b30"
    except Exception as e:
        log_debug(f"MEM_PRESSURE psutil fallback EXCEPTION: {e}")
        return "—", "#888888"


def _get_memory_pressure_psutil():
    """Fallback when CLI unavailable: estimate pressure from psutil RAM metrics."""
    return _get_memory_pressure_fallback()


def _get_ram_usage():
    """Return RAM percentage, total GB, available GB."""
    if psutil is None:
        return "—", "—", "—"
    mem = psutil.virtual_memory()
    return mem.percent, mem.total / (1024**3), mem.available / (1024**3)


# ──────────────────────────────────────────────
# GPU Monitoring — cross-platform (Apple Silicon + Linux NVIDIA/AMD)
# ──────────────────────────────────────────────

def _get_gpu_info():
    """Return dict with GPU utilization and temperature.
    
    Works on:
      - macOS Apple Silicon: uses powermetrics for GPU/Neural Engine stats
      - Linux NVIDIA: uses nvidia-smi for util + temp
      - Linux AMD: tries rocm-smi or falls back to /sys/class/drm
      - Any platform without GPU support: returns "—", "—"
    
    Returns: (utilization_str, temperature_str)
    """
    platform_name = sys.platform
    
    # ── macOS Apple Silicon ──────────────────────────────
    if platform_name == "darwin":
        return _get_gpu_macos()
    
    # ── Linux NVIDIA ─────────────────────────────────────
    if platform_name == "linux":
        util, temp = _get_gpu_linux_nvidia()
        if util is not None:  # Found nvidia-smi and working
            return util, temp
        
        # Try AMD ROCm
        util, temp = _get_gpu_linux_amd_rocm()
        if util is not None:
            return util, temp
    
    # ── No GPU tooling found ─────────────────────────────
    return "—", "—"


def _get_gpu_macos():
    """Get GPU stats on macOS via powermetrics CLI.
    
    powermetrics (macOS built-in) reports:
      - GPU load in percent
      - GPU temperature in Celsius
    
    Returns: (utilization_str, temperature_str) or ("—", "—") if unavailable.
    """
    try:
        # Run powermetrics once for a short sample to get GPU stats
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "-i", "100", "--samplers", "gpu"],
            capture_output=True, text=True, timeout=5
        )
        
        raw = result.stdout + result.stderr
        
        # Parse GPU load: look for "GPU load: XX.XX%" or similar
        import re
        gpu_load_match = re.search(r'GPU\s+load:\s*([\d.]+)%', raw)
        if not gpu_load_match:
            # Try alternate format from newer macOS versions
            gpu_load_match = re.search(r'gpu\s+utilization:\s*([\d.]+)%', raw, re.IGNORECASE)
        
        util_str = f"{float(gpu_load_match.group(1)):.0f}%" if gpu_load_match else None
        
        # Parse GPU temperature: look for "GPU die temp:" or "Temperature:"
        temp_match = re.search(r'GPU\s+die\s+temp:\s*([\d.]+)\s*C', raw, re.IGNORECASE)
        if not temp_match:
            temp_match = re.search(r'(?:GPU\s+|Board\s+)?temp(?:erature)?[:\s]+([\d.]+)\s*[Cc]', raw, re.IGNORECASE)
        
        temp_str = f"{float(temp_match.group(1)):.0f}°C" if temp_match else None
        
        # If powermetrics required sudo and failed, try without sudo (may not work)
        if util_str is None:
            result2 = subprocess.run(
                ["powermetrics", "-i", "100", "--samplers", "gpu"],
                capture_output=True, text=True, timeout=5
            )
            raw2 = result2.stdout + result2.stderr
            gpu_load_match2 = re.search(r'GPU\s+load:\s*([\d.]+)%', raw2)
            temp_match2 = re.search(r'GPU\s+die\s+temp:\s*([\d.]+)\s*C', raw2, re.IGNORECASE)
            
            if gpu_load_match2:
                util_str = f"{float(gpu_load_match2.group(1)):.0f}%"
            if temp_match2:
                temp_str = f"{float(temp_match2.group(1)):.0f}°C"
        
        return (util_str or "—", temp_str or "—")
    
    except FileNotFoundError:
        # powermetrics not found — definitely not macOS, or very old OS
        log_debug("GPU: powermetrics CLI not found")
        return "—", "—"
    except subprocess.TimeoutExpired:
        log_debug("GPU: powermetrics timed out (may need sudo)")
        return "—", "—"
    except Exception as e:
        log_debug(f"GPU macos exception: {type(e).__name__}: {e}")
        return "—", "—"


def _get_gpu_linux_nvidia():
    """Get GPU stats on Linux via nvidia-smi.
    
    Returns: (utilization_str, temperature_str) or (None, None) if no NVIDIA GPU found.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            return None, None
        
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        if not lines:
            return None, None
        
        # nvidia-smi may report multiple GPUs; show the first one (or aggregate)
        utils = []
        temps = []
        name_parts = []
        
        for line in lines[:4]:  # Up to 4 GPUs max
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                try:
                    utils.append(float(parts[0]))
                    temps.append(float(parts[1]))
                    name_parts.append(parts[2])
                except (ValueError, IndexError):
                    pass
        
        if not utils:
            return None, None
        
        # Use first GPU as primary, note others in detail
        util_str = f"{utils[0]:.0f}%"
        
        if len(utils) > 1:
            util_str += f" (total {len(utils)} GPUs)"
        
        temp_str = f"{temps[0]:.0f}°C"
        
        # If multiple GPUs, mention it
        if len(temps) > 1:
            temps_list = ", ".join(f"{t:.0f}°C" for t in temps)
            temp_str += f" ({', '.join(name_parts[:2])})"
        
        return util_str, temp_str
    
    except FileNotFoundError:
        # nvidia-smi not found — no NVIDIA GPU tooling
        log_debug("GPU: nvidia-smi not found")
        return None, None
    except subprocess.TimeoutExpired:
        log_debug("GPU: nvidia-smi timed out")
        return None, None
    except Exception as e:
        log_debug(f"GPU linux_nvidia exception: {type(e).__name__}: {e}")
        return None, None


def _get_gpu_linux_amd_rocm():
    """Get GPU stats on Linux via rocm-smi.
    
    Returns: (utilization_str, temperature_str) or (None, None) if no AMD GPU found.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showinfo"],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            return None, None
        
        raw = result.stdout + result.stderr
        
        import re
        
        # Parse GPU load
        gpu_load_match = re.search(r'(?:GPU\s+)?(?:average)?\s*load:\s*([\d.]+)%', raw, re.IGNORECASE)
        
        util_str = f"{float(gpu_load_match.group(1)):.0f}%" if gpu_load_match else None
        
        # Parse temperature — may be "edge" or "junction" temp
        temp_match = re.search(r'(?:GPU\s+)?(?:edge|junction)\s+temp:\s*([\d.]+)\s*C', raw, re.IGNORECASE)
        
        temp_str = f"{float(temp_match.group(1)):.0f}°C" if temp_match else None
        
        return (util_str or "—", temp_str or "—")
    
    except FileNotFoundError:
        log_debug("GPU: rocm-smi not found")
        return None, None
    except subprocess.TimeoutExpired:
        log_debug("GPU: rocm-smi timed out")
        return None, None
    except Exception as e:
        log_debug(f"GPU linux_amd_rocm exception: {type(e).__name__}: {e}")
        return None, None


# ──────────────────────────────────────────────
# HTML generation — responsive mobile dashboard
# ──────────────────────────────────────────────

def generate_html(pressure, pressure_color, ram_pct, ram_total, ram_avail, 
                  gpu_util, gpu_temp, lm_online, lm_gen_speed, lm_detail, lm_ttft, lm_prompt_tps, lm_context):
    timestamp = datetime.now().strftime("%H:%M:%S")
    dot_color = "#34c759" if lm_online else "#ff3b30"
    
    # Handle empty/no data context (after reset, no logs yet, or placeholder "—")
    if not lm_context or lm_context == "—" or isinstance(lm_context, str):
        context_display = "Reset — waiting for completions..."
    else:
        context_display = f"{lm_context:.0f} tokens"
    commit_hash, commit_ts = _get_git_info()
    uptime = _uptime_str()
    logs_enabled = _cache.get("logs_enabled", False)
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
  .status-bar .commit-ts {{ color: #888; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85em; }}
  .status-bar .uptime {{ color: #34c759; }}
  .debug-toggle {{ position: fixed; bottom: 80px; right: 20px; background: {dbg_color}; color: white; border: none; padding: 10px; border-radius: 50%; font-size: 16px; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.5); opacity: 0.7; }}
  .debug-toggle:hover {{ opacity: 1; }}
  /* Update button styling */
  .update-btn {{ position: fixed; bottom: 140px; right: 20px; background: #007aff; color: white; border: none; padding: 10px; border-radius: 50%; font-size: 16px; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.5); opacity: 0.7; }}
  .update-btn:hover {{ opacity: 1; }}
  .forward-btn {{ position: fixed; bottom: 220px; right: 20px; background: #0a84ff; color: white; border: none; padding: 10px; border-radius: 50%; font-size: 16px; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.5); opacity: 0.7; }}
  .forward-btn:hover {{ opacity: 1; }}
  .update-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
  /* Model info button styling */
  .info-btn {{ position: fixed; bottom: 180px; right: 20px; background: #5856d6; color: white; border: none; padding: 10px; border-radius: 50%; font-size: 16px; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.5); opacity: 0.7; }}
  .info-btn:hover {{ opacity: 1; }}
  .info-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
  /* Model info popup */
  .info-popup {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #2d2d2d; border-radius: 14px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); z-index: 1000; min-width: 300px; max-width: 400px; }}
  .info-popup h3 {{ margin: 0 0 16px; color: #fff; font-size: 1.1em; }}
  .info-popup .info-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #3d3d3d; }}
  .info-popup .info-label {{ color: #888; font-size: 0.9em; }}
  .info-popup .info-value {{ color: #fff; font-weight: 600; }}
  .info-popup .close-btn {{ position: absolute; top: 12px; right: 12px; background: none; border: none; color: #888; font-size: 1.2em; cursor: pointer; }}
  .info-popup .close-btn:hover {{ color: #fff; }}
  .info-overlay {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 999; }}
  /* GPU card styling */
  .gpu-util {{ color: #bf5af2; }}
  .gpu-temp {{ font-size: 1.3em; font-weight: 600; }}
  /* Compact layout */
  body {{ padding: 20px 16px 80px; }}
  .card {{ padding: 12px 16px; margin-bottom: 10px; }}
  .value {{ font-size: 1.8em; }}
  .ttft-value {{ font-size: 1.4em; }}
  h2 {{ font-size: 0.75em; margin-bottom: 4px; }}
  .sub {{ font-size: 0.8em; }}
  .header h1 {{ font-size: 1.2em; margin-bottom: 16px; }}
  .footer {{ margin-top: 20px; font-size: 0.7em; }}
  /* Mobile responsive */
  @media (max-width: 600px) {{
    body {{ padding: 8px 10px 90px; }}
    .card {{ padding: 12px 14px; margin-bottom: 10px; border-radius: 10px; }}
    .value {{ font-size: 1.4em; }}
    .ttft-value {{ font-size: 1.1em; }}
    h2 {{ font-size: 0.7em; margin-bottom: 4px; letter-spacing: 0.5px; }}
    .sub {{ font-size: 0.75em; line-height: 1.3; }}
    .header h1 {{ font-size: 1.0em; margin-bottom: 12px; }}
    .footer {{ font-size: 0.65em; margin-top: 16px; }}
    .pressure-badge {{ font-size: 0.95em; padding: 5px 12px; }}
    .status-dot {{ width: 7px; height: 7px; margin-right: 3px; }}
    
    /* Compact status bar for mobile */
    .status-bar {{ flex-wrap: wrap; gap: 8px; padding: 6px 10px; font-size: 0.7em; }}
    .status-bar span {{ font-size: 0.9em; }}
    
    /* Stack floating buttons vertically on left side */
    .refresh-btn, .update-btn, .forward-btn, .info-btn, .debug-toggle {{
      right: auto;
      left: 10px;
      bottom: auto;
      padding: 8px;
      font-size: 14px;
    }}
    .refresh-btn {{ bottom: 10px; }}
    .debug-toggle {{ bottom: 65px; }}
    .update-btn {{ bottom: 120px; }}
    .info-btn {{ bottom: 175px; }}
    .forward-btn {{ bottom: 230px; }}
  }}
  
  /* Small mobile (iPhone SE, etc.) */
  @media (max-width: 375px) {{
    body {{ padding: 6px 8px 80px; }}
    .card {{ padding: 10px 12px; margin-bottom: 8px; }}
    .value {{ font-size: 1.2em; }}
    .ttft-value {{ font-size: 1em; }}
    h2 {{ font-size: 0.6em; }}
    .sub {{ font-size: 0.65em; }}
    .header h1 {{ font-size: 0.95em; }}
    .footer {{ font-size: 0.55em; }}
    .status-bar {{ font-size: 0.65em; }}
    .pressure-badge {{ font-size: 0.85em; }}
  }}
  /* Capture button styling */
  .capture-btn {{ position: fixed; bottom: 220px; right: 20px; background: #30d158; color: white; border: none; padding: 10px; border-radius: 50%; font-size: 16px; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.5); opacity: 0.7; }}
  .capture-btn:hover {{ opacity: 1; }}
  .capture-btn.active {{ background: #ff453a; opacity: 1; }}
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
    <div class="sub">Avg time LM Studio spent processing prompt tokens before generation</div>
  </div>

  <div class="footer">
    Last updated: {timestamp} · LM Studio API probe every {CACHE_TTL}s<br>
    Auto-refresh page with ↻ button below
  </div>

  <div class="status-bar">
    <span class="commit">{commit_hash}</span>
    <span class="commit-ts">{commit_ts}</span>
    <span>·</span>
    <span>{commit_age} ago</span>
    <span>·</span>
    <span class="uptime">{uptime}</span>
    <span>·</span>
    <span>● running</span>
  </div>

  <button class="debug-toggle" id="debugBtn" title="Toggle debug logging" onclick="toggleDebug()">&#x1F41B;</button>
  <button class="info-btn" id="infoBtn" title="Show model info (free, no inference)" onclick="showModelInfo()">&#x1F4CB;</button>
  <button class="update-btn" id="updateBtn" title="Update from GitHub" onclick="updateServer()">&#x1F504;</button>
  <button class="forward-btn" id="forwardBtn" title="Forward logs for debugging" onclick="forwardLogs()">&#x1F4E4;</button>
  <button class="stop-btn" id="stopBtn" title="Stop dashboard and log server" onclick="stopServer()">&#x1F6D1;</button>

  <script>
    // HTML escape helper
    function escapeHtml(str) {{
      const div = document.createElement('div');
      div.appendChild(document.createTextNode(str));
      return div.innerHTML;
    }}

    // Toggle debug logging on/off
    function toggleDebug() {{
      const btn = document.getElementById('debugBtn');
      const isOn = btn.dataset.state === 'on';
      fetch('/debug/toggle?enable=' + (isOn ? '0' : '1'))
        .then(() => {{ location.reload(); }})
        .catch(() => {{}});
    }}

    // Show model info popup
    function showModelInfo() {{
      const btn = document.getElementById('infoBtn');
      btn.disabled = true;
      btn.innerHTML = '&#x23F3;';
      
      // Create overlay and popup
      const overlay = document.createElement('div');
      overlay.className = 'info-overlay';
      overlay.onclick = closeInfoPopup;
      document.body.appendChild(overlay);
      
      const popup = document.createElement('div');
      popup.className = 'info-popup';
      popup.innerHTML = '<button class="close-btn" onclick="closeInfoPopup()">&times;</button><h3>📋 LM Studio Model Info</h3><div id="infoContent">Loading...</div>';
      document.body.appendChild(popup);
      
      fetch('/api/lm_info')
        .then(response => response.json())
        .then(data => {{
          const content = document.getElementById('infoContent');
          if (data.error) {{
            content.innerHTML = '<div class="info-row"><span class="info-label">Status</span><span class="info-value" style="color: #ff3b30;">❌ </span>' + escapeHtml(data.error) + '</div>';
          }} else {{
            let html = '';
            if (data.name) html += '<div class="info-row"><span class="info-label">Model</span><span class="info-value">' + escapeHtml(data.name) + '</span></div>';
            if (data.context_length && data.context_length !== 'Unknown') {{
              const ctxNum = parseInt(data.context_length);
              const ctxPercent = data.context_length !== 'Unknown' ? ((128000 / ctxNum) * 100).toFixed(1) : '—';
              html += '<div class="info-row"><span class="info-label">Max Context</span><span class="info-value">' + escapeHtml(data.context_length) + ' tokens (' + ctxPercent + '% of 128k)</span></div>';
            }}
            if (data.object) html += '<div class="info-row"><span class="info-label">Type</span><span class="info-value">' + escapeHtml(data.object) + '</span></div>';
            if (data.online) html += '<div class="info-row"><span class="info-label">Status</span><span class="info-value" style="color: #34c759;">✅ Online</span></div>';
            if (!data.online) html += '<div class="info-row"><span class="info-label">Status</span><span class="info-value" style="color: #ff3b30;">❌ Offline</span></div>';
            content.innerHTML = html;
          }}
          btn.innerHTML = '&#x1F4CB;';
          btn.disabled = false;
        }})
        .catch(error => {{
          const content = document.getElementById('infoContent');
          content.innerHTML = '<div class="info-row"><span class="info-label">Error</span><span class="info-value" style="color: #ff3b30;">❌ </span>' + escapeHtml(error.message) + '</div>';
          btn.innerHTML = '&#x1F4CB;';
          btn.disabled = false;
        }});
    }}
    
    function closeInfoPopup() {{
      const overlay = document.querySelector('.info-overlay');
      const popup = document.querySelector('.info-popup');
      if (overlay) overlay.remove();
      if (popup) popup.remove();
    }}

    // Update from GitHub
    function updateServer() {{
      const btn = document.getElementById('updateBtn');
      btn.disabled = true;
      btn.innerHTML = '&#x23F3;'; // hourglass
      btn.title = 'Updating...';
      
      fetch('/api/update')
        .then(response => response.json())
        .then(data => {{
          if (data.status === 'success') {{
            btn.innerHTML = '&#x2705;'; // checkmark
            btn.title = 'Update successful!';
            setTimeout(() => {{ location.reload(); }}, 1000);
          }} else {{
            btn.innerHTML = '&#x274C;'; // X
            btn.title = 'Update failed: ' + data.error;
            setTimeout(() => {{ 
              btn.innerHTML = '&#x1F504;';
              btn.disabled = false;
            }}, 2000);
          }}
        }})
        .catch(error => {{
          btn.innerHTML = '&#x274C;';
          btn.title = 'Update failed: ' + error.message;
          setTimeout(() => {{ 
            btn.innerHTML = '&#x1F504;';
            btn.disabled = false;
          }}, 2000);
        }});
    }}

    // Forward logs for debugging
    function forwardLogs() {{
      const btn = document.getElementById('forwardBtn');
      btn.innerHTML = '&#x23F3;'; // hourglass
      btn.title = 'Fetching logs...';
      
      fetch('/api/log_forward')
        .then(response => response.json())
        .then(data => {{
          if (data.error) {{
            alert('❌ ' + data.error);
            btn.innerHTML = '&#x1F4E4;';
            btn.title = 'Forward logs for debugging';
          }} else {{
            const logContent = document.createElement('div');
            logContent.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#1c1c1e;color:#fff;padding:20px;border-radius:10px;max-width:80%;max-height:70%;overflow:auto;font-family:monospace;font-size:12px;z-index:10000;box-shadow:0 4px 20px rgba(0,0,0,0.8)';
            logContent.innerHTML = '<div style="margin-bottom:10px;font-weight:bold;font-size:14px;">📋 Log Forward — ' + data.file + '</div><pre style="white-space:pre-wrap;word-break:break-all;">' + escapeHtml(data.content) + '</pre><div style="margin-top:10px;text-align:right;"><button onclick="this.parentElement.parentElement.remove()" style="background:#0a84ff;color:white;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;">Close</button></div>';
            document.body.appendChild(logContent);
            btn.innerHTML = '&#x1F4E4;';
            btn.title = 'Forward logs for debugging';
          }}
        }})
        .catch(error => {{
          alert('❌ Failed to fetch logs: ' + error.message);
          btn.innerHTML = '&#x1F4E4;';
          btn.title = 'Forward logs for debugging';
        }});
    }}

    // Stop dashboard and log server
    function stopServer() {{
      if (confirm('⚠️ Are you sure you want to stop the dashboard and log server?')) {{
        const btn = document.getElementById('stopBtn');
        btn.disabled = true;
        btn.innerHTML = '&#x23F3;';
        btn.title = 'Stopping...';
        
        fetch('/api/stop')
          .then(response => response.json())
          .then(data => {{
            if (data.status === 'stopping') {{
              btn.innerHTML = '&#x2705;';
              btn.title = 'Stopped!';
              setTimeout(() => {{
                alert('✅ Dashboard and log server stopped successfully');
                location.reload();
              }}, 500);
            }}
          }})
          .catch(error => {{
            btn.innerHTML = '&#x274C;';
            btn.title = 'Stop failed: ' + error.message;
            setTimeout(() => {{
              btn.innerHTML = '&#x1F6D1;';
              btn.disabled = false;
            }}, 2000);
          }});
      }}
    }}

    // Stop dashboard and log server
    function stopServer() {{
      if (confirm('⚠️ Are you sure you want to stop the dashboard and log server?')) {{
        const btn = document.getElementById('stopBtn');
        btn.disabled = true;
        btn.innerHTML = '&#x23F3;';
        btn.title = 'Stopping...';
        
        fetch('/api/stop')
          .then(response => response.json())
          .then(data => {{
            if (data.status === 'stopping') {{
              btn.innerHTML = '&#x2705;';
              btn.title = 'Stopped!';
              setTimeout(() => {{
                alert('✅ Dashboard and log server stopped successfully');
                location.reload();
              }}, 500);
            }}
          }})
          .catch(error => {{
            btn.innerHTML = '&#x274C;';
            btn.title = 'Stop failed: ' + error.message;
            setTimeout(() => {{
              btn.innerHTML = '&#x1F6D1;';
              btn.disabled = false;
            }}, 2000);
          }});
      }}
    }}

    // Reset aggregation window
    function resetAvg() {{
      if (!confirm('Reset the running average of last {AVG_WINDOW} requests? This will clear cached stats and re-scan logs.')) return;
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
</html>}}"""


# ──────────────────────────────────────────────
# HTTP server handler
# ──────────────────────────────────────────────

class Handler(http.server.SimpleHTTPRequestHandler):
    allow_reuse_address = True
    
    def do_GET(self):
        # Auto-reload: if the script file changed since we started, restart ourselves
        global _SCRIPT_MTIME_START
        current_mtime = os.path.getmtime(_SCRIPT_PATH)
        if current_mtime > _SCRIPT_MTIME_START:
            print("🔄 Script changed — reloading…")
            os.execv(sys.executable, [sys.executable, __file__])

        if self.path == "/":
            try:
                pressure, p_color = _get_memory_pressure()
                ram_pct, ram_total, ram_avail = _get_ram_usage()
                gpu_util, gpu_temp = _get_gpu_info()
                lm_online, lm_gen_speed, lm_detail, lm_ttft, lm_prompt_tps, lm_context = _get_cached_lm_stats()

                html = generate_html(
                    pressure, p_color, ram_pct, ram_total, ram_avail,
                    gpu_util, gpu_temp,
                    lm_online, lm_gen_speed, lm_detail, lm_ttft, lm_prompt_tps, lm_context
                )
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())
            except BrokenPipeError:
                pass  # Client disconnected, ignore
            except Exception as e:
                print(f"❌ ERROR serving /: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
                try:
                    err_html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Error</title>
<style>*{{box-sizing:border-box;margin:0;padding:24px;font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#1a1a1a;color:#fff}}
body{{max-width:600px;margin:auto;padding-top:40px}}h1{{font-size:1.5em;margin-bottom:16px}}pre{{background:#2d2d2d;padding:12px;border-radius:8px;overflow-x:auto;font-size:0.8em;white-space:pre-wrap}}
.sub{{color:#aaa;margin-top:16px}}</style></head><body>
<h1>⚠️ Dashboard Error</h1><pre>{traceback.format_exc()}</pre>
<div class="sub">Check ~/llm-monitor/logs/crash.log for details. Server process is alive (status endpoint works).</div>
</body></html>}}"""
                    self.send_response(500)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(err_html.encode())
                except Exception:
                    pass  # Connection may already be closed

        elif self.path == "/status":
            uptime = int(time.time() - _START_TIME)
            gpu_util, gpu_temp = _get_gpu_info()
            # Fetch fresh LM Studio stats (lazy — probes on first request, then caches for CACHE_TTL)
            lm_online, lm_gen_speed, lm_detail, lm_ttft, lm_prompt_tps, lm_context = _get_cached_lm_stats()
            
            status_data = {
                "status": "running",
                "uptime_seconds": uptime,
                "pid": os.getpid(),
                "timestamp": datetime.now().isoformat(),
                "gpu": {
                    "utilization": gpu_util,
                    "temperature": gpu_temp,
                },
                "lm_stats": {
                    "online": lm_online,
                    "gen_speed": lm_gen_speed,
                    "detail": lm_detail,
                    "ttft_ms": lm_ttft,
                    "prompt_tps": lm_prompt_tps,
                    "context_size": lm_context,
                },
            }
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status_data, indent=2).encode())

        elif self.path == "/api/rollback":
            # Manual rollback endpoint
            backup_path = _get_latest_backup()
            if not backup_path:
                self.send_response(404)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No backups found"}).encode())
                return
            
            if _restore_backup(backup_path):
                log_debug("Manual rollback successful")
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success",
                    "message": f"Restored from {os.path.basename(backup_path)}",
                    "backup_file": backup_path,
                }).encode())
            else:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Restore failed"}).encode())
        
        elif self.path == "/api/update":
            # Self-update: git pull + validate + swap + restart
            # This is the main entry point for the "Update" button
            log_debug("🔄 Update initiated via /api/update")
            
            # Step 1: Create backup of current script
            backup_path = _create_backup()
            if not backup_path:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "failed",
                    "error": "Backup creation failed",
                }).encode())
                return
            
            # Step 2: Try git pull
            try:
                result = subprocess.run(
                    ["git", "pull", "origin", "main"],
                    capture_output=True, text=True, timeout=30,
                    cwd=os.path.dirname(_SCRIPT_PATH) or "."
                )
                
                if result.returncode != 0:
                    log_debug(f"git pull failed: {result.stderr}")
                    # Rollback if pull fails
                    _restore_backup(backup_path)
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "failed",
                        "error": f"git pull failed: {result.stderr[:200]}",
                    }).encode())
                    return
                
                log_debug(f"git pull successful: {result.stdout[:100]}")
            except subprocess.TimeoutExpired:
                log_debug("git pull timed out")
                _restore_backup(backup_path)
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "failed",
                    "error": "git pull timed out",
                }).encode())
                return
            except Exception as e:
                log_debug(f"git pull error: {type(e).__name__}: {e}")
                _restore_backup(backup_path)
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "failed",
                    "error": f"git pull error: {type(e).__name__}",
                }).encode())
                return
            
            # Step 3: Validate the new script
            if not _validate_script():
                log_debug("Validation failed after git pull")
                _restore_backup(backup_path)
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "failed",
                    "error": "Script validation failed after update",
                }).encode())
                return
            
            # Step 4: Cleanup old backups
            _cleanup_old_backups()
            
            # Step 5: Signal restart (we'll restart after sending response)
            log_debug("✅ Update successful — restarting server...")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "success",
                "message": "Update successful — restarting server...",
                "backup_created": backup_path,
            }).encode())
            
            # Restart the server in a new process
            time.sleep(1)  # Give client time to receive response
            os.execv(sys.executable, [sys.executable, __file__])
        
        elif self.path == "/api/backup/list":
            # List available backups
            try:
                if not os.path.exists(BACKUP_DIR):
                    backups = []
                else:
                    backups = sorted([
                        f for f in os.listdir(BACKUP_DIR) 
                        if f.startswith("llm_monitor_") and f.endswith(".py")
                    ])
                
                backup_info = []
                for b in backups:
                    path = os.path.join(BACKUP_DIR, b)
                    stat = os.stat(path)
                    backup_info.append({
                        "filename": b,
                        "size_bytes": stat.st_size,
                        "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "backups": backup_info,
                    "count": len(backup_info),
                }).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        
        elif self.path == "/reset_avg":
            # Reset aggregation window: clear all metrics and mark fresh-start so we skip log scanning
            _cache["recent_requests"] = []
            for key in list(_cache.keys()):
                if key.startswith("lm_"):
                    del _cache[key]
            print("🔄 Aggregation reset — cleared all stats; dashboard will show empty until new completions arrive")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "reset"}).encode())

        elif self.path == "/api/health":
            # Health check endpoint
            is_valid = _validate_script()
            latest_backup = _get_latest_backup()
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "healthy" if is_valid else "unhealthy",
                "script_valid": is_valid,
                "has_backup": latest_backup is not None,
                "backup_path": latest_backup,
                "uptime_seconds": int(time.time() - _START_TIME),
            }).encode())

        elif self.path == "/api/log_forward":
            # Forward log content to dashboard for debugging parsing issues
            log_files = _find_log_files()
            if not log_files:
                self.send_response(404)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No log files found"}).encode())
                return
            
            newest = max(log_files, key=os.path.getmtime)
            with open(newest, "r", errors="replace") as f:
                lines = f.readlines()[-100:]
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "file": os.path.basename(newest),
                "content": "".join(lines)
            }).encode())
        
        elif self.path == "/api/stop":
            # Stop dashboard and log server gracefully
            log_debug("Stop endpoint called — shutting down dashboard and watchdog")
            
            # Stop watchdog thread
            _stop_watchdog()
            
            # Kill log server
            pid_file = os.path.join(LOGS_DIR, "log_server.pid")
            if os.path.exists(pid_file):
                try:
                    with open(pid_file, "r") as f:
                        pid = int(f.read().strip())
                    os.kill(pid, signal.SIGTERM)
                    log_debug(f"Log server (PID {pid}) terminated")
                except (ValueError, ProcessLookupError, PermissionError):
                    pass
                os.remove(pid_file)
            
            # Shutdown the server
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "stopping"}).encode())
            
            # Schedule shutdown after response
            threading.Thread(target=lambda: httpd.shutdown(), daemon=True).start()
        
        elif self.path == "/api/watchdog":
            # Manual watchdog check — returns status dict
            status = _handle_watchdog_check()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode())
        
        elif self.path == "/api/watchdog/status":
            # Just return status without checking/restarting
            stale = _check_stale_pid()
            alive = _check_log_server_alive()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "watchdog_active": _watchdog_running,
                "log_server_alive": alive,
                "pid_stale": stale,
            }, indent=2).encode())
        
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
                "recent_probes": len(_cache.get("recent_probes", [])),
                "lm_studio_url": LM_STUDIO_URL,
                "probe_method": "API (no log parsing)",
                "logs": logs,
            }, indent=2).encode())

        else:
            super().do_GET()


# ──────────────────────────────────────────────
# Entry point
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

def _generate_log_html(filename: str) -> str:
    """Generate HTML page for viewing a log file."""
    filepath = os.path.join(LOGS_DIR, filename)
    if not os.path.exists(filepath):
        return "<h2>File not found</h2>"
    
    with open(filepath, 'r', errors='replace') as f:
        content = f.read()
    
    # Escape HTML
    content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Highlight errors/warnings
    import re
    content = re.sub(r'(ERROR|Exception|Traceback)', r'<span class="error">\1</span>', content)
    content = re.sub(r'(WARNING|WARN)', r'<span class="warning">\1</span>', content)
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{filename} - Log Viewer</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'SF Mono', monospace; margin: 0; padding: 20px; background: #1a1a1a; color: #d4d4d4; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
            .header h1 {{ margin: 0; font-size: 1.5em; }}
            .header a {{ color: #4fc3f7; text-decoration: none; }}
            .header a:hover {{ text-decoration: underline; }}
            .stats {{ font-size: 0.9em; color: #888; }}
            #log-content {{ background: #0d0d0d; padding: 15px; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; line-height: 1.6; font-size: 13px; }}
            .error {{ color: #ff6b6b; font-weight: bold; }}
            .warning {{ color: #ffd93d; }}
            .timestamp {{ color: #4fc3f7; }}
            input[type="text"] {{ width: 300px; padding: 8px; background: #2d2d2d; border: 1px solid #444; color: #fff; border-radius: 4px; }}
            .search-box {{ margin-bottom: 15px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📄 {filename}</h1>
            <div>
                <a href="/">← Back to logs</a>
            </div>
        </div>
        <div class="stats">
            {os.path.getsize(filepath)} bytes | Last modified: {os.path.getmtime(filepath):.0f}
        </div>
        <div class="search-box">
            <input type="text" id="search" placeholder="Search logs..." oninput="filterLogs()">
        </div>
        <div id="log-content">{content}</div>
        <script>
            function filterLogs() {{
                const search = document.getElementById('search').value.toLowerCase();
                const lines = document.getElementById('log-content').textContent.split('\\n');
                const filtered = lines.filter(line => !search || line.toLowerCase().includes(search));
                document.getElementById('log-content').textContent = filtered.join('\\n');
            }}
        </script>
    </body>
    </html>
    """


def _generate_logs_index_html() -> str:
    """Generate HTML index page listing all log files."""
    files_html = []
    for f in sorted(os.listdir(LOGS_DIR), reverse=True):
        if f.endswith('.log') and not f.endswith('.pid'):
            filepath = os.path.join(LOGS_DIR, f)
            size = os.path.getsize(filepath)
            size_kb = f"{size / 1024:.1f} KB"
            mtime = os.path.getmtime(filepath)
            mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
            files_html.append(f'<li><a href="/logs/{f}">📄 {f}</a> <span class="stats">{size_kb} | {mtime_str}</span></li>')
    
    if not files_html:
        files_html = '<li>No log files found</li>'
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Log Viewer - LM Monitor</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'SF Mono', monospace; margin: 0; padding: 20px; background: #1a1a1a; color: #d4d4d4; }}
            h1 {{ color: #4fc3f7; }}
            ul {{ list-style: none; padding: 0; }}
            li {{ padding: 10px; margin: 5px 0; background: #2d2d2d; border-radius: 6px; display: flex; justify-content: space-between; align-items: center; }}
            a {{ color: #4fc3f7; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .stats {{ color: #888; font-size: 0.9em; }}
            .back-link {{ display: inline-block; margin-bottom: 20px; color: #4fc3f7; text-decoration: none; }}
            .back-link:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <a href="/" class="back-link">← LM Monitor Dashboard</a>
        <h1>📋 Available Log Files</h1>
        <ul>
            {"".join(files_html)}
        </ul>
    </body>
    </html>
    """


class LogServerHandler(BaseHTTPRequestHandler):
    """Custom HTTP handler for log server."""
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(_generate_logs_index_html().encode())
        elif self.path.startswith('/logs/'):
            filename = self.path.split('/logs/')[-1]
            if filename.endswith('.log'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(_generate_log_html(filename).encode())
            else:
                self.send_error(404, "Not found")
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def _check_log_server_alive() -> bool:
    """Check if log server is responding on port 8081."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("127.0.0.1", 8081))
        sock.close()
        if result != 0:
            return False
        # Try to actually connect and get a response
        sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock2.settimeout(2)
        sock2.connect(("127.0.0.1", 8081))
        sock2.sendall(b"GET / HTTP/1.0\r\n\r\n")
        response = sock2.recv(1024)
        sock2.close()
        return b"200" in response or b"Log" in response or len(response) > 0
    except Exception:
        return False


def _check_stale_pid() -> bool:
    """Check if PID file exists but process is dead."""
    pid_file = os.path.join(LOGS_DIR, "log_server.pid")
    if not os.path.exists(pid_file):
        return False
    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        # Check if process is actually running
        os.kill(pid, 0)  # Signal 0 checks existence without sending signal
        return False  # Process is alive
    except (ValueError, ProcessLookupError, PermissionError):
        # PID is stale
        try:
            os.remove(pid_file)
            log_debug("Removed stale PID file")
        except:
            pass
        return True


def _restart_log_server() -> None:
    """Kill existing log server and start a new one."""
    # Kill existing process if any
    pid_file = os.path.join(LOGS_DIR, "log_server.pid")
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            log_debug(f"Killed existing log server (PID {pid})")
            time.sleep(1)  # Wait for clean shutdown
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        try:
            os.remove(pid_file)
        except:
            pass
    
    # Start fresh
    _start_log_server()
    log_debug("Log server restarted")


def _log_server_watchdog() -> None:
    """Watchdog that checks log server health every 30 seconds."""
    log_debug("Log server watchdog started (checks every 30s)")
    while True:
        time.sleep(30)
        if _check_stale_pid():
            log_debug("Stale PID detected — restarting log server")
            _restart_log_server()
        elif not _check_log_server_alive():
            log_debug("Log server not responding — restarting")
            _restart_log_server()
        else:
            log_debug("Log server healthy")


def _start_log_server() -> None:
    """Start a background HTTP server on port 8081 serving logs with HTML interface."""
    import socket
    
    # Check if already running on port 8081
    if _check_log_server_alive():
        log_debug("Log server already running on port 8081")
        return
    
    # Not running — start it detached from the dashboard process
    pid_file = os.path.join(LOGS_DIR, "log_server.pid")
    proc = subprocess.Popen(
        [sys.executable, "-c", f"""
import sys
sys.path.insert(0, {repr(os.path.dirname(os.path.abspath(__file__)))})
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer
from llm_monitor import LogServerHandler
import os

LOGS_DIR = {repr(LOGS_DIR)}

class ReusableLogServer(TCPServer):
    allow_reuse_address = True

with ReusableLogServer(("0.0.0.0", 8081), LogServerHandler) as httpd:
    print("Log server running on port 8081", flush=True)
    httpd.serve_forever()
""",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Save PID so stop.sh can kill it cleanly
    with open(pid_file, "w") as f:
        f.write(str(proc.pid))
    log_debug(f"Log server started on port 8081 (PID {proc.pid})")


# ──────────────────────────────────────────────
# Log server watchdog — background thread
# ──────────────────────────────────────────────

_watchdog_running = False
_watchdog_thread = None


def _start_watchdog() -> None:
    """Start the watchdog thread that monitors log server health."""
    global _watchdog_running, _watchdog_thread
    if _watchdog_running:
        log_debug("Watchdog already running")
        return
    
    _watchdog_running = True
    _watchdog_thread = threading.Thread(target=_log_server_watchdog, daemon=True)
    _watchdog_thread.start()
    log_debug("Watchdog thread started")


def _stop_watchdog() -> None:
    """Stop the watchdog thread."""
    global _watchdog_running
    _watchdog_running = False
    log_debug("Watchdog thread stopped")


def _handle_watchdog_check() -> dict:
    """Manual watchdog check — returns status dict."""
    stale = _check_stale_pid()
    alive = _check_log_server_alive()
    
    status = {
        "watchdog_active": _watchdog_running,
        "log_server_alive": alive,
        "pid_stale": stale,
        "pid_file": os.path.join(LOGS_DIR, "log_server.pid"),
        "pid_exists": os.path.exists(os.path.join(LOGS_DIR, "log_server.pid")),
    }
    
    if stale or not alive:
        log_debug("Manual watchdog check — restarting log server")
        _restart_log_server()
        status["action_taken"] = "restarted"
    else:
        status["action_taken"] = "none — server healthy"
    
    return status



# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True
    allow_reuse_port = True


if __name__ == "__main__":
    print(f"🚀 Dashboard running at http://<YOUR_MAC_IP>:{PORT}")
    print(f"   Probing LM Studio API: {LM_STUDIO_URL}/v1/chat/completions")
    print(f"   Probe every {CACHE_TTL}s · Running avg over last {AVG_WINDOW} probes")
    
    # Show GPU detection status
    gpu_util, gpu_temp = _get_gpu_info()
    print(f"   GPU: {gpu_util} utilization / {gpu_temp} temperature")
    
    print("Press Ctrl+C to stop.")
    
    # Auto-rollback: if script is invalid, restore from backup
    if not _validate_script():
        print("⚠️  Script validation failed — attempting rollback...")
        backup_path = _get_latest_backup()
        if backup_path:
            print(f"   Restoring from backup: {os.path.basename(backup_path)}")
            if _restore_backup(backup_path):
                print("✅ Rollback successful — restarting with backup...")
                os.execv(sys.executable, [sys.executable, __file__])
            else:
                print("❌ Rollback failed — continuing with broken script")
        else:
            print("❌ No backups available — continuing with broken script")
    
    # Seed cache with initial status
    _cache.update({
        "lm_online": False,
        "lm_gen_speed": "—",
        "lm_detail": f"Waiting for first log scan...",
        "lm_ttft": "—",
        "lm_ts": time.time(),
    })
    print("✅ Initial cache seeded — waiting for log data")
    
    # Start the background log server (survives dashboard crashes)
    _start_log_server()
    print(f"📂 Log server started on port 8081 — I can read crash/debug logs remotely")
    
    # Watchdog disabled to prevent crash loop
    # _start_watchdog()
    print("🐕 Watchdog disabled (was causing restart loop)")
    
    try:
        httpd = ReusableTCPServer(("", PORT), Handler)
        httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
