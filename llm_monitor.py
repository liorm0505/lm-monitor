#!/usr/bin/env python3
"""
LM Monitor — Real-time LLM Inference Dashboard for Mac Mini (Apple Silicon)
=============================================================================

Monitors:
  • macOS Memory Pressure (Low / Medium / High)
  • Unified RAM Usage (GB used / available / total)
  • GPU Utilization & Temperature (macOS Apple Silicon + Linux NVIDIA/AMD fallback)
  • LM Studio Avg Generation Speed (tokens/sec via API probe)
  • LM Studio Avg Prompt Processing Time (via API probe)

Uses lightweight HTTP probes to LM Studio's /v1/chat/completions endpoint 
with max_tokens=1 — zero log parsing, no verbose logging required.

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
# Backup management — timestamped script backups for rollback
# ──────────────────────────────────────────────

def _create_backup():
    """Create a timestamped backup of the current script.
    
    Returns backup path or None on failure.
    """
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"llm_monitor_{timestamp}.py")
        
        # Copy current script to backup
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
        
        # Remove oldest if we have more than MAX_BACKUPS
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
_SCRIPT_PATH = os.path.abspath(__file__)
_SCRIPT_MTIME_START = os.path.getmtime(_SCRIPT_PATH)


# ──────────────────────────────────────────────
# LM Studio API probe — lightweight stats collection
# ──────────────────────────────────────────────

def _probe_lm_studio():
    """Send a single max_tokens=1 probe to /v1/chat/completions and extract timing stats.
    
    Returns dict with: gen_speed_tps (float), ttft_ms (float), or None on failure.
    This is the ONLY data collection path now — no log parsing at all.
    """
    # urllib is always available on Python 3.9+ (stdlib)

    probe_payload = {
        "model": "",  # empty string tells LM Studio to use the currently loaded model
        "messages": [{"role": "user", "content": "."}],
        "max_tokens": 1,
        "temperature": 0.1,
    }
    
    url = f"{LM_STUDIO_URL}/v1/chat/completions"
    data = json.dumps(probe_payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        start_time = time.time()
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed_s = time.time() - start_time
        
        # Extract stats from response
        stats = body.get("stats") or {}  # LM Studio native API
        
        gen_speed = None
        ttft_ms = None
        
        # 1. Try LM Studio native stats format
        if "tokens_per_second" in stats:
            tps = stats["tokens_per_second"]
            if tps > 0:
                gen_speed = tps
        
        if "time_to_first_token_seconds" in stats:
            ttft_s = stats["time_to_first_token_seconds"]
            ttft_ms = ttft_s * 1000

        # 2. Fallback: calculate from OpenAI-compatible response + timing
        if gen_speed is None or ttft_ms is None:
            usage = body.get("usage", {}) or {}
            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0
            total_tokens = usage.get("total_tokens", 0) or 0
            
            # Calculate generation speed from elapsed time
            if completion_tokens > 0 and elapsed_s > 0:
                gen_speed = completion_tokens / elapsed_s
            
            # Estimate TTFT: time minus generation time for remaining tokens
            if completion_tokens > 0 and elapsed_s > 0 and prompt_tokens > 0:
                # Simple estimate: TTFT ≈ total_time - (total_tokens / gen_speed)
                # But since we only have 1 completion token, TTFT ≈ elapsed time
                ttft_ms = elapsed_s * 1000
        
        return {"gen_speed_tps": gen_speed, "ttft_ms": ttft_ms}

    except urllib.error.URLError as e:
        # LM Studio not running or unreachable
        log_debug(f"PROBE FAILED: {e}")
        return None
    except Exception as e:
        log_debug(f"PROBE ERROR: {type(e).__name__}: {e}")
        return None


def _probe_lm_studio_batch(n=3):
    """Send n probes quickly and collect individual results.
    
    Returns list of probe result dicts (may contain None for failed probes).
    """
    results = []
    for _ in range(n):
        r = _probe_lm_studio()
        results.append(r)
        # Small delay between probes so stats are independent
        time.sleep(0.15)
    return results


# ──────────────────────────────────────────────
# LM Studio model info — free metadata probe (no inference)
# ──────────────────────────────────────────────

def get_lm_studio_info():
    """Fetch model metadata from LM Studio without triggering inference.
    
    Calls GET /v1/models which returns model name, context_length, etc.
    Zero token cost — pure metadata.
    
    Returns dict with: name, context_length, object type, or error message.
    """
    try:
        url = f"{LM_STUDIO_URL}/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        models = data.get("data", [])
        if models:
            m = models[0]
            ctx = m.get("context_length")
            return {
                "name": m.get("id", "Unknown"),
                "context_length": ctx if ctx else "Unknown",
                "object": m.get("object", "model"),
                "online": True,
            }
        return {"error": "No models loaded in LM Studio", "online": True}
    except urllib.error.URLError as e:
        return {"error": f"LM Studio unreachable: {e}", "online": False}
    except Exception as e:
        return {"error": f"Failed to fetch model info: {type(e).__name__}: {e}", "online": False}


# ──────────────────────────────────────────────
# Cache state — prevents frequent API calls on page reloads
# ──────────────────────────────────────────────
_cache = {
    "lm_online": False,
    "lm_gen_speed": "—",       # Avg generation speed (tokens/sec) from probe results
    "lm_detail": "Waiting...", # Detail string with avg metrics
    "lm_ttft": "—",            # Avg time to first token (ms) from probes
    "lm_ts": 0,
    "logs_enabled": False,     # Toggle: capture stdout/stderr to /debug/logs
    "recent_probes": [],       # Last N probe results (in memory)
}


# ──────────────────────────────────────────────
# Data collection — lightweight API probes
# ──────────────────────────────────────────────

def _get_cached_lm_stats():
    """Return cached LM Studio stats unless TTL has expired.
    
    On cache miss: sends 3 quick max_tokens=1 probes, averages results.
    No file I/O, no log parsing — pure HTTP.
    """
    now = time.time()
    if now - _cache["lm_ts"] > CACHE_TTL:
        # Send probe batch
        print(f"📡 Probing LM Studio at {LM_STUDIO_URL}...")
        
        probes = _probe_lm_studio_batch(3)
        success_count = sum(1 for p in probes if p is not None)
        
        if success_count == 0:
            # All probes failed — mark offline
            print("⚠️  LM Studio unreachable at this time")
            _cache.update({
                "lm_online": False,
                "lm_gen_speed": "—",
                "lm_detail": f"LM Studio not reachable at {LM_STUDIO_URL}",
                "lm_ttft": "—",
                "lm_ts": now,
            })
        else:
            # Aggregate successful probes
            gen_speeds = [p["gen_speed_tps"] for p in probes if p is not None and p.get("gen_speed_tps")]
            ttfts = [p["ttft_ms"] for p in probes if p is not None and p.get("ttft_ms") is not None]
            
            avg_gen = f"{sum(gen_speeds) / len(gen_speeds):.1f} tok/s" if gen_speeds else "—"
            avg_ttft = f"{sum(ttfts) / len(ttfts):.0f} ms" if ttfts else "—"
            
            # Store individual probe results for running average display
            _cache["recent_probes"] = probes[-AVG_WINDOW:]
            
            detail_parts = [
                f"{success_count}/3 probes succeeded",
                f"TTFT: {avg_ttft}",
                f"Gen: {avg_gen}",
            ]
            
            _cache.update({
                "lm_online": True,
                "lm_gen_speed": avg_gen if gen_speeds else "—",
                "lm_detail": ", ".join(detail_parts),
                "lm_ttft": avg_ttft if ttfts else "—",
                "lm_ts": now,
            })
            
            print(f"📊 Probe results: {success_count}/3 succeeded")
    
    return (
        _cache["lm_online"],
        _cache["lm_gen_speed"],
        _cache["lm_detail"],
        _cache["lm_ttft"],
        "—",   # lm_prompt_tps — not tracked with API probe approach
        0,     # lm_context_size — not tracked with API probe approach
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
  * {{ box-sizing: border-box; margin: 0; padding: 24px; }}
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
  /* Update button styling */
  .update-btn {{ position: fixed; bottom: 140px; right: 20px; background: #007aff; color: white; border: none; padding: 10px; border-radius: 50%; font-size: 16px; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.5); opacity: 0.7; }}
  .update-btn:hover {{ opacity: 1; }}
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
    <h2><span class="status-dot"></span>GPU Utilization</h2>
    <div class="value gpu-util">{gpu_util}</div>
    <div class="sub">{gpu_temp} · Apple Silicon Neural Engine + GPU (Metal)</div>
  </div>

  <div class="card">
    <h2><span class="status-dot"></span>Avg Generation Speed</h2>
    <div class="value">{lm_gen_speed}</div>
    <div class="sub">{lm_detail}</div>
  </div>

  <div class="card">
    <h2><span class="status-dot"></span>Avg Time to First Token (TTFT)</h2>
    <div class="ttft-value">{lm_ttft}</div>
    <div class="sub">Average time from request start until first output token arrives</div>
  </div>

  <div class="footer">
    Last updated: {timestamp} · LM Studio API probe every {CACHE_TTL}s<br>
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

  <button class="debug-toggle" id="debugBtn" title="Toggle debug logging" onclick="toggleDebug()">&#x1F41B;</button>

  <button class="info-btn" id="infoBtn" title="Show model info (free, no inference)" onclick="showModelInfo()">&#x1F4CB;</button>
  <button class="update-btn" id="updateBtn" title="Update from GitHub" onclick="updateServer()">&#x1F504;</button>

  <script>
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
            content.innerHTML = `<div class="info-row"><span class="info-label">Status</span><span class="info-value" style="color: #ff3b30;">❌ ${data.error}</span></div>`;
          }} else {{
            let html = '';
            if (data.name) html += `<div class="info-row"><span class="info-label">Model</span><span class="info-value">${data.name}</span></div>`;
            if (data.context_length && data.context_length !== 'Unknown') {{
              const ctxNum = parseInt(data.context_length);
              const ctxPercent = data.context_length !== 'Unknown' ? ((128000 / ctxNum) * 100).toFixed(1) : '—';
              html += `<div class="info-row"><span class="info-label">Max Context</span><span class="info-value">${data.context_length} tokens (${ctxPercent}% of 128k)</span></div>`;
            }}
            if (data.object) html += `<div class="info-row"><span class="info-label">Type</span><span class="info-value">${data.object}</span></div>`;
            if (data.online) html += `<div class="info-row"><span class="info-label">Status</span><span class="info-value" style="color: #34c759;">✅ Online</span></div>`;
            if (!data.online) html += `<div class="info-row"><span class="info-label">Status</span><span class="info-value" style="color: #ff3b30;">❌ Offline</span></div>`;
            content.innerHTML = html;
          }}
          btn.innerHTML = '&#x1F4CB;';
          btn.disabled = false;
        }})
        .catch(error => {{
          const content = document.getElementById('infoContent');
          content.innerHTML = `<div class="info-row"><span class="info-label">Error</span><span class="info-value" style="color: #ff3b30;">❌ ${error.message}</span></div>`;
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

    // Reset aggregation window
    function resetAvg() {{
      if (!confirm('Reset the running average of last {AVG_WINDOW} requests? This will clear cached stats and re-probe.')) return;
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
</body></html>"""
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
        
        elif self.path == "/api/lm_info":
            # On-demand: fetch LM Studio model metadata (free, no inference)
            info = get_lm_studio_info()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(info, indent=2).encode())

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

        elif self.path == "/reset_avg":
            # Reset aggregation window: clear all metrics and mark fresh-start so we skip API calls
            _cache["recent_probes"] = []
            for key in list(_cache.keys()):
                if key.startswith("lm_"):
                    del _cache[key]
            print("🔄 Aggregation reset — cleared all stats; dashboard will show empty until new probes complete")
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
    
    # Seed cache with one startup probe (no warning — LM Studio may start later)
    probe_result = _probe_lm_studio()
    if probe_result and probe_result.get("gen_speed_tps"):
        print(f"✅ LM Studio API reachable — gen speed: {probe_result['gen_speed_tps']:.1f} tok/s")
        _cache.update({
            "lm_online": True,
            "lm_gen_speed": f"{probe_result['gen_speed_tps']:.1f} tok/s",
            "lm_detail": "Initial probe — gen speed from startup test",
            "lm_ttft": f"{probe_result.get('ttft_ms', 0):.0f} ms" if probe_result.get("ttft_ms") else "—",
            "lm_ts": time.time(),
        })
    else:
        # Don't warn — LM Studio often starts after the monitor. Just seed cache offline.
        _cache.update({
            "lm_online": False,
            "lm_gen_speed": "—",
            "lm_detail": f"LM Studio not reachable at {LM_STUDIO_URL} (will re-probe every {CACHE_TTL}s)",
            "lm_ttft": "—",
            "lm_ts": time.time(),
        })
    
    # Start the background log server (survives dashboard crashes)
    _start_log_server()
    print(f"📂 Log server started on port 8081 — I can read crash/debug logs remotely")
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass
