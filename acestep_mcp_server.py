#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mcp[cli]>=1.0.0",
#   "httpx>=0.27.0",
# ]
# ///
"""
ACE-Step MCP Server
stdio transport — spawned on demand by Claude Code.
Boots acestep-api as a subprocess, waits for it to be ready,
then proxies MCP tool calls to the REST API at localhost:PORT.

Environment variables (set in .claude.json env block):
  ACESTEP_DIR          Path to ACE-Step repo root (default: directory of this file)
  PORT                 API server port (default: 8001)
  ACESTEP_LM_BACKEND   mlx | vllm | pt (default: mlx for Apple Silicon)
  ACESTEP_DEVICE       mps | cuda | cpu (default: mps)
  ACESTEP_LM_MODEL_PATH  LM model name (default: acestep-5Hz-lm-1.7B)
  ACESTEP_CONFIG_PATH    DiT variant (default: acestep-v15-turbo)
  ACESTEP_API_KEY      Optional API key for the server
  ACESTEP_START_TIMEOUT  Seconds to wait for server startup (default: 180)
"""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
ACESTEP_DIR = Path(os.environ.get("ACESTEP_DIR", Path(__file__).parent))
PORT = int(os.environ.get("PORT", "8001"))
API_BASE = f"http://127.0.0.1:{PORT}"
API_KEY = os.environ.get("ACESTEP_API_KEY", "")
START_TIMEOUT = int(os.environ.get("ACESTEP_START_TIMEOUT", "180"))
POLL_INTERVAL = 2.0
MAX_POLL_SECONDS = 600

mcp = FastMCP("acestep")

# Subprocess handle — set when we boot the server ourselves
_api_process: Optional[subprocess.Popen] = None


def _auth_headers() -> dict:
    if API_KEY:
        return {"Authorization": f"Bearer {API_KEY}"}
    return {}


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

async def _is_api_running() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{API_BASE}/health", timeout=3.0)
            return r.status_code == 200
    except Exception:
        return False


def _find_uv() -> str:
    """Locate the uv binary, checking common macOS/Linux install paths."""
    import shutil
    uv = shutil.which("uv")
    if uv:
        return uv
    for candidate in [
        os.path.expanduser("~/.local/bin/uv"),
        "/opt/homebrew/bin/uv",
        "/usr/local/bin/uv",
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise RuntimeError("uv not found. Install it: https://docs.astral.sh/uv/")


async def _boot_api_server() -> None:
    """Start acestep-api as a background subprocess and wait for it to be ready."""
    global _api_process

    uv = _find_uv()
    env = {**os.environ, "PORT": str(PORT)}
    log_path = Path(ACESTEP_DIR) / "acestep_mcp_boot.log"

    cmd = [uv, "run", "acestep-api"]

    with open(log_path, "w") as log:
        _api_process = subprocess.Popen(
            cmd,
            cwd=str(ACESTEP_DIR),
            env=env,
            stdout=log,
            stderr=log,
        )

    deadline = time.time() + START_TIMEOUT
    while time.time() < deadline:
        await asyncio.sleep(3.0)
        if _api_process.poll() is not None:
            raise RuntimeError(
                f"ACE-Step API process exited prematurely (code {_api_process.returncode})"
            )
        if await _is_api_running():
            return

    _api_process.terminate()
    raise RuntimeError(f"ACE-Step API did not become ready within {START_TIMEOUT}s")


async def ensure_api_running() -> None:
    """Ensure the ACE-Step API is reachable, booting it if necessary."""
    if await _is_api_running():
        return
    await _boot_api_server()


def _shutdown_api() -> None:
    """Terminate the API subprocess if we started it."""
    global _api_process
    if _api_process and _api_process.poll() is None:
        _api_process.terminate()
        try:
            _api_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _api_process.kill()
        _api_process = None


# Graceful shutdown on SIGTERM/SIGINT
for _sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_sig, lambda *_: (_shutdown_api(), sys.exit(0)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _poll_task(client: httpx.AsyncClient, task_id: str) -> dict:
    """Poll /query_result until the task finishes. Returns the raw result dict."""
    deadline = time.time() + MAX_POLL_SECONDS
    while time.time() < deadline:
        await asyncio.sleep(POLL_INTERVAL)
        r = await client.post(
            f"{API_BASE}/query_result",
            json={"task_id_list": [task_id]},
            headers=_auth_headers(),
            timeout=10.0,
        )
        r.raise_for_status()
        result = r.json()["data"][0]
        outer_status = result["status"]

        # Check outer status codes
        if outer_status >= 3:  # error / timeout
            raise RuntimeError(result.get("progress_text", "Generation failed"))

        # The outer status often stays at 1 even after completion.
        # Check the inner result items for stage="succeeded" as the real done signal.
        if outer_status >= 1 and result.get("result"):
            try:
                items = json.loads(result["result"])
                if items and all(item.get("stage") == "succeeded" for item in items):
                    return result
            except (json.JSONDecodeError, KeyError):
                pass

        if outer_status == 2:  # fallback: explicit done status
            return result

    raise RuntimeError(f"Task {task_id} timed out after {MAX_POLL_SECONDS}s")


def _collect_outputs(result: dict, out_dir: Path, audio_format: str) -> list[str]:
    """Copy generated files to out_dir with human-readable names. Returns path list."""
    from urllib.parse import urlparse, parse_qs, unquote
    items = json.loads(result["result"])
    task_id = result["task_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, item in enumerate(items):
        # file field is a URL like /v1/audio?path=%2FUsers%2F...
        # Extract and decode the actual filesystem path from the query string
        parsed = urlparse(item["file"])
        qs = parse_qs(parsed.query)
        if "path" in qs:
            src = Path(unquote(qs["path"][0]))
        else:
            src = Path(item["file"])  # fallback: treat as direct path
        seed_used = item.get("seed_value", "unknown")
        dest = out_dir / f"{task_id}_{i}_seed{seed_used}.{audio_format}"
        shutil.copy2(src, dest)
        paths.append(str(dest))
    return paths


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_music(
    prompt: str,
    duration: float = 120.0,
    lyrics: str = "[instrumental]",
    bpm: int = 72,
    key: str = "C minor",
    seed: int = -1,
    num_samples: int = 1,
    guidance_scale: float = 10.0,
    steps: int = 20,
    audio_format: str = "wav",
    output_path: Optional[str] = None,
) -> dict:
    """
    Generate music using ACE-Step. Returns a list of output file paths.

    Defaults are tuned for UPTIME (72 BPM, C minor, 120s loops, WAV for Audacity).

    Args:
        prompt:         Style/mood/instrument description.
                        e.g. "dark ambient, drone pads, sub-bass pulse, clinical cold"
        duration:       Length in seconds. Use 120 for UPTIME loops.
        lyrics:         Lyrics text, or "[instrumental]" for no vocals.
        bpm:            Beats per minute. Must match exactly across all layers (default 72).
        key:            Musical key e.g. "C minor", "A major" (default "C minor").
        seed:           Random seed for reproducibility. -1 = random.
                        Use fixed seeds (1, 2, 3) to generate comparable variants.
        num_samples:    Number of variants to generate in one call (1–8).
                        Returns an array of paths — generate 3, pick the best.
        guidance_scale: Prompt adherence (default 10.0 = tight; 7.0 = more creative).
        steps:          Inference steps. 20 = clean loops; 8 = fast draft; 50 = high quality.
        audio_format:   Output format. "wav" for Audacity handoff; "ogg" for final Roblox export.
        output_path:    Directory to write files. Defaults to ./acestep_output/
    """
    await ensure_api_running()

    out_dir = Path(output_path) if output_path else Path.cwd() / "acestep_output"

    payload = {
        "prompt": prompt,
        "lyrics": lyrics,
        "audio_duration": duration,
        "bpm": bpm,
        "key_scale": key,
        "seed": seed,
        "use_random_seed": seed == -1,
        "batch_size": num_samples,
        "guidance_scale": guidance_scale,
        "inference_steps": steps,
        "audio_format": audio_format,
        "task_type": "text2music",
    }

    async with httpx.AsyncClient() as client:
        # Submit
        r = await client.post(
            f"{API_BASE}/release_task",
            json=payload,
            headers=_auth_headers(),
            timeout=15.0,
        )
        r.raise_for_status()
        resp = r.json()
        task_id = resp["data"]["task_id"]

        # Wait
        result = await _poll_task(client, task_id)

    output_paths = _collect_outputs(result, out_dir, audio_format)

    return {
        "success": True,
        "task_id": task_id,
        "output_paths": output_paths,
        "num_generated": len(output_paths),
        "bpm": bpm,
        "key": key,
        "duration": duration,
        "steps": steps,
        "guidance_scale": guidance_scale,
    }


@mcp.tool()
async def format_music_prompt(
    prompt: str,
    lyrics: str = "",
    temperature: float = 0.85,
) -> dict:
    """
    Use ACE-Step's LLM to enhance and structure a music prompt before generation.
    Returns suggested caption, lyrics, BPM, key, time signature, duration, and vocal language.

    Useful when you have a rough idea and want the model to fill in musical details.

    Args:
        prompt:      Rough description e.g. "tense ambient for a datacenter game"
        lyrics:      Optional starting lyrics (leave empty for instrumental ideas)
        temperature: LLM creativity (0.0 = deterministic, 1.0 = creative, default 0.85)
    """
    await ensure_api_running()

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{API_BASE}/format_input",
            json={"prompt": prompt, "lyrics": lyrics, "temperature": temperature},
            headers=_auth_headers(),
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json().get("data", {})


@mcp.tool()
async def list_models() -> dict:
    """List available ACE-Step DiT and LM models, showing which are loaded."""
    await ensure_api_running()

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{API_BASE}/v1/models",
            headers=_auth_headers(),
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json().get("data", {})


@mcp.tool()
async def get_server_health() -> dict:
    """
    Check ACE-Step server status — whether it's running, which models are loaded,
    and current queue depth.
    """
    if not await _is_api_running():
        return {"status": "offline", "message": "ACE-Step API is not running"}

    async with httpx.AsyncClient() as client:
        health = (await client.get(f"{API_BASE}/health", timeout=5.0)).json()
        stats = (
            await client.get(
                f"{API_BASE}/v1/stats", headers=_auth_headers(), timeout=5.0
            )
        ).json().get("data", {})

    return {**health, "stats": stats}


@mcp.tool()
async def load_lora(lora_path: str, adapter_name: Optional[str] = None) -> dict:
    """
    Load a LoRA adapter for style fine-tuning.

    Args:
        lora_path:    Path to the LoRA weights directory or file.
        adapter_name: Optional name to identify this adapter.
    """
    await ensure_api_running()

    payload = {"lora_path": lora_path}
    if adapter_name:
        payload["adapter_name"] = adapter_name

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{API_BASE}/v1/lora/load",
            json=payload,
            headers=_auth_headers(),
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json().get("data", {})


@mcp.tool()
async def set_lora_strength(scale: float, adapter_name: Optional[str] = None) -> dict:
    """
    Set LoRA adapter strength (0.0 = no effect, 1.0 = full effect).

    Args:
        scale:        Strength between 0.0 and 1.0.
        adapter_name: Specific adapter to adjust (optional).
    """
    await ensure_api_running()

    payload = {"scale": scale}
    if adapter_name:
        payload["adapter_name"] = adapter_name

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{API_BASE}/v1/lora/scale",
            json=payload,
            headers=_auth_headers(),
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json().get("data", {})


@mcp.tool()
async def unload_lora() -> dict:
    """Unload the currently active LoRA adapter."""
    await ensure_api_running()

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{API_BASE}/v1/lora/unload",
            headers=_auth_headers(),
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json().get("data", {})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
