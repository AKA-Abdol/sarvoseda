"""Diagnose slow dataset downloads.

Answers three questions, in order, because the fix is different for each:

1. **Is the server's link slow at all?**  Measured against a neutral CDN
   (Cloudflare). If this is slow, nothing about Hugging Face is the problem.
2. **Is Hugging Face slower than that baseline?**  If yes, it is HF or the
   route to it, not your machine.
3. **Does opening more connections help?**  This is the decisive one. A single
   TCP stream on a long-haul, high-latency route is throughput-limited by the
   bandwidth-delay product long before the link itself saturates. If 8
   connections give roughly 8x, the link has headroom and parallel downloading
   is the fix. If they give ~1x, the pipe really is full and only a mirror,
   a closer region, or patience will help.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

#: Neutral CDN endpoints, tried in order - some reject unknown clients.
BASELINE_URLS = [
    "https://speed.cloudflare.com/__down?bytes={n}",
    "https://ash-speed.hetzner.com/100MB.bin",
    "https://proof.ovh.net/files/100Mb.dat",
]
UA = "Mozilla/5.0 (compatible; vcprep-netcheck/1.0)"
CHUNK = 32 * 1024 * 1024        # 32 MB probe


def _session(token: Optional[str] = None):
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = UA
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def _timed_range(session, url: str, start: int, length: int) -> Tuple[int, float]:
    """Fetch one byte range; return (bytes received, seconds)."""
    headers = {"Range": f"bytes={start}-{start + length - 1}"}
    began = time.time()
    got = 0
    with session.get(url, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for block in resp.iter_content(chunk_size=1 << 20):
            got += len(block)
    return got, time.time() - began


def _mbps(byte_count: int, seconds: float) -> float:
    return (byte_count / (1024 * 1024)) / seconds if seconds > 0 else 0.0


def baseline(size: int = CHUNK) -> float:
    """Raw link throughput against a neutral CDN, in MB/s."""
    session = _session()
    for template in BASELINE_URLS:
        url = template.format(n=size)
        began = time.time()
        got = 0
        try:
            with session.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                for block in resp.iter_content(chunk_size=1 << 20):
                    got += len(block)
                    if got >= size:      # endpoints with a fixed size
                        break
        except Exception as exc:
            log.debug("baseline %s failed: %s", url.split("/")[2], exc)
            continue
        if got > 0:
            return _mbps(got, time.time() - began)
    log.warning("every neutral-CDN baseline probe failed")
    return float("nan")


def hf_single(url: str, token: Optional[str], size: int = CHUNK) -> float:
    session = _session(token)
    got, seconds = _timed_range(session, url, 0, size)
    return _mbps(got, seconds)


def hf_parallel(url: str, token: Optional[str], connections: int = 8,
                size: int = CHUNK) -> float:
    """Throughput with ``connections`` simultaneous ranges over the same span."""
    session = _session(token)
    per = size // connections
    began = time.time()
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [pool.submit(_timed_range, session, url, i * per, per)
                   for i in range(connections)]
        total = sum(f.result()[0] for f in futures)
    return _mbps(total, time.time() - began)


def supports_ranges(url: str, token: Optional[str]) -> bool:
    """Whether the server honours Range - required for parallel downloads."""
    session = _session(token)
    try:
        got, _ = _timed_range(session, url, 0, 1024)
        return got == 1024
    except Exception:
        return False


def run(url: str, token: Optional[str] = None, connections: int = 8,
        size: int = CHUNK, skip_baseline: bool = False) -> Dict[str, float]:
    results: Dict[str, float] = {}

    if not skip_baseline:
        print(f"  probing neutral CDN baseline ({size // (1<<20)} MB) ...")
        results["baseline"] = baseline(size)

    print(f"  probing Hugging Face, 1 connection ({size // (1<<20)} MB) ...")
    results["hf_single"] = hf_single(url, token, size)

    if supports_ranges(url, token):
        print(f"  probing Hugging Face, {connections} connections ...")
        results["hf_parallel"] = hf_parallel(url, token, connections, size)
        results["connections"] = connections
    else:
        print("  server does not honour Range requests - skipping parallel probe")

    return results


def report(results: Dict[str, float], total_gb: float = 35.0) -> None:
    """Print the measurements and say what to actually do about them."""
    base = results.get("baseline", float("nan"))
    single = results.get("hf_single", float("nan"))
    par = results.get("hf_parallel")
    conns = int(results.get("connections", 0))

    print()
    print(f"{'probe':<34} {'MB/s':>9}   {'35 GB would take':>18}")
    print("-" * 66)

    def row(label: str, value: float) -> None:
        if value != value:                       # NaN
            print(f"{label:<34} {'failed':>9}")
            return
        hours = (total_gb * 1024) / value / 3600 if value > 0 else float("inf")
        eta = f"{hours:.1f} h" if hours >= 1 else f"{hours * 60:.0f} min"
        print(f"{label:<34} {value:9.2f}   {eta:>18}")

    if base == base:
        row("neutral CDN (Cloudflare)", base)
    row("Hugging Face, 1 connection", single)
    if par is not None:
        row(f"Hugging Face, {conns} connections", par)

    print()
    print("verdict:")

    if base == base and base < 5 and single < 5:
        print("  * The server's own link is slow, not Hugging Face -")
        print("    the neutral CDN is just as slow. Nothing in this repo can")
        print("    fix that; you need a better-connected machine or region.")
    elif base == base and single < base * 0.5:
        print("  * Hugging Face is markedly slower than the neutral CDN, so it")
        print("    is HF or the route to it - not your link.")

    if par is not None and single > 0:
        gain = par / single
        if gain >= 2.0:
            print(f"  * {conns} connections gave {gain:.1f}x. Your link has "
                  f"headroom and a single")
            print("    stream is the bottleneck (bandwidth-delay product).")
            print("    FIX:  vcprep run --download-mode parallel "
                  f"--connections {conns}")
        else:
            print(f"  * {conns} connections gave only {gain:.1f}x - the pipe is "
                  f"genuinely full.")
            print("    Parallel downloading will not help. Options: run the")
            print("    pipeline in a region closer to HF, or set HF_ENDPOINT to")
            print("    a mirror.")
    print()
