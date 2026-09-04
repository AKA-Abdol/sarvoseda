"""NISQA v2 adapter - multidimensional speech quality.

NISQA predicts MOS plus four interpretable dimensions:

    noi  noisiness        (overlaps DNSMOS BAK)
    col  coloration       (spectral damage from separation)
    dis  discontinuity    <- the reason NISQA is here
    loud loudness

**Discontinuity** is the axis DNSMOS does not have. MDX-Net separation fails
by warbling, chopping and time-smearing the vocal, and that is exactly what
``dis_pred`` measures. On a dataset of movie audio pushed through Kim Vocal 2
it is the single most useful artifact detector available.

NISQA is a separate repository under an academic/non-commercial licence, so it
is not vendored. ``scripts/setup_nisqa.sh`` clones it, and this adapter drives
its batch predictor as a subprocess - far more robust across NISQA versions
than importing its internals.
"""
from __future__ import annotations

import csv
import logging
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

#: NISQA CSV column -> our manifest metric name
_COLUMNS = {
    "mos_pred": "nisqa_mos",
    "noi_pred": "nisqa_noi",
    "dis_pred": "nisqa_dis",
    "col_pred": "nisqa_col",
    "loud_pred": "nisqa_loud",
}


class Nisqa:
    """Batch scorer. NISQA has meaningful startup cost, so it runs per-shard
    over a whole directory rather than per file."""

    def __init__(self, repo_dir: str, weights: str, device: str = "auto",
                 batch_size: int = 10, num_workers: int = 0,
                 timeout_per_file: float = 2.0, timeout_floor: float = 300.0):
        self.repo = Path(repo_dir)
        self.weights = Path(weights)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        #: A wedged subprocess would otherwise stall the pipeline forever, and
        #: with output captured there is nothing on screen to show why. Budget
        #: generously - this is a deadlock guard, not a performance limit.
        self.timeout_per_file = timeout_per_file
        self.timeout_floor = timeout_floor

        script = self.repo / "run_predict.py"
        if not script.exists():
            raise FileNotFoundError(
                f"NISQA not found at {self.repo}. Install it with:\n"
                f"    bash scripts/setup_nisqa.sh\n"
                f"or disable it with  quality.use_nisqa: false"
            )
        if not self.weights.exists():
            raise FileNotFoundError(f"NISQA weights not found at {self.weights}")

    # ------------------------------------------------------------------
    def score_dir(self, audio_dir: str) -> Dict[str, Dict[str, float]]:
        """Score every wav in ``audio_dir``. Keys are file *stems*."""
        audio_dir = str(Path(audio_dir).resolve())
        n_files = sum(1 for _ in Path(audio_dir).glob("*.wav"))
        timeout = max(self.timeout_floor, n_files * self.timeout_per_file)
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable, "run_predict.py",
                "--mode", "predict_dir",
                "--pretrained_model", str(self.weights.resolve()),
                "--data_dir", audio_dir,
                "--num_workers", str(self.num_workers),
                "--bs", str(self.batch_size),
                "--output_dir", tmp,
            ]
            log.info("NISQA: scoring %d file(s), timeout %.0fs", n_files, timeout)
            log.debug("NISQA: %s", " ".join(cmd))
            started = time.time()
            try:
                proc = subprocess.run(cmd, cwd=str(self.repo),
                                      capture_output=True, text=True,
                                      timeout=timeout)
            except subprocess.TimeoutExpired:
                log.error("NISQA exceeded %.0fs on %d file(s) and was killed. "
                          "Continuing without its scores - DNSMOS and the "
                          "heuristics still apply. Disable it with "
                          "quality.use_nisqa: false, or raise "
                          "timeout_per_file.", timeout, n_files)
                return {}
            except OSError as exc:
                log.error("could not run NISQA: %s", exc)
                return {}
            if proc.returncode != 0:
                log.error("NISQA failed (%s): %s", proc.returncode,
                          proc.stderr[-2000:])
                return {}
            log.info("NISQA finished in %.1fs", time.time() - started)
            return _parse_results(Path(tmp))


def _parse_results(out_dir: Path) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    candidates = list(out_dir.glob("*.csv"))
    if not candidates:
        log.error("NISQA produced no CSV in %s", out_dir)
        return results
    with open(candidates[0], newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            deg = row.get("deg") or row.get("filename") or ""
            if not deg:
                continue
            stem = Path(deg).stem
            metrics: Dict[str, float] = {}
            for src, dst in _COLUMNS.items():
                if row.get(src) not in (None, ""):
                    try:
                        metrics[dst] = float(row[src])
                    except ValueError:
                        pass
            if metrics:
                results[stem] = metrics
    return results
