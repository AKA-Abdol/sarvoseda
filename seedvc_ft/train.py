"""Drive seed-vc v2's ``train_v2.py`` via accelerate.

Wrapping rather than reimplementing: seed-vc's training loop is the upstream
project's business, and reproducing it here would drift the moment upstream
changes. This module builds the command, checks the preconditions that
otherwise fail twenty minutes in, and streams the output.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .config import FinetuneConfig, TrainConfig

log = logging.getLogger(__name__)


def build_command(cfg: FinetuneConfig, dataset_dir: Optional[str] = None
                  ) -> List[str]:
    train: TrainConfig = cfg.train
    data_dir = dataset_dir or str(Path(cfg.data.dataset_dir) / "train")

    cmd: List[str] = ["accelerate", "launch"]
    if train.num_processes > 1:
        cmd += ["--multi_gpu", "--num_processes", str(train.num_processes)]
    if train.mixed_precision and train.mixed_precision != "no":
        cmd += ["--mixed_precision", train.mixed_precision]
    cmd += ["--main_process_port", str(train.main_process_port)]

    cmd += [
        "train_v2.py",
        "--config", train.config,
        "--dataset-dir", str(Path(data_dir).resolve()),
        "--run-name", train.run_name,
        "--batch-size", str(train.batch_size),
        "--max-steps", str(train.max_steps),
        "--max-epochs", str(train.max_epochs),
        "--save-every", str(train.save_every),
        "--num-workers", str(train.num_workers),
    ]
    if train.train_cfm:
        cmd.append("--train-cfm")
    if train.train_ar:
        cmd.append("--train-ar")
    if train.pretrained_cfm_ckpt:
        cmd += ["--pretrained-cfm-ckpt", train.pretrained_cfm_ckpt]
    if train.pretrained_ar_ckpt:
        cmd += ["--pretrained-ar-ckpt", train.pretrained_ar_ckpt]
    cmd += list(train.extra_args)
    return cmd


def preflight(cfg: FinetuneConfig, dataset_dir: Optional[str] = None) -> None:
    """Fail fast and legibly rather than deep inside someone else's stack."""
    repo = Path(cfg.train.repo_dir)
    script = repo / "train_v2.py"
    if not script.exists():
        raise FileNotFoundError(
            f"seed-vc not found at {repo}. Install it with:\n"
            f"    bash scripts/setup_seedvc.sh"
        )

    config_path = repo / cfg.train.config
    if not config_path.exists():
        raise FileNotFoundError(
            f"training config not found: {config_path}\n"
            f"(--config is resolved relative to the seed-vc repo)"
        )

    data_dir = Path(dataset_dir or Path(cfg.data.dataset_dir) / "train")
    if not data_dir.exists():
        raise FileNotFoundError(
            f"dataset directory not found: {data_dir}\nRun `vcft prepare` first."
        )
    audio = [p for p in data_dir.rglob("*")
             if p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg"}]
    if not audio:
        raise RuntimeError(f"no audio files in {data_dir}")
    log.info("dataset: %d files in %s", len(audio), data_dir)

    if shutil.which("accelerate") is None:
        raise RuntimeError(
            "`accelerate` is not on PATH. Install seed-vc's requirements:\n"
            f"    pip install -r {repo / 'requirements.txt'} accelerate"
        )

    if not (cfg.train.train_cfm or cfg.train.train_ar):
        raise ValueError("nothing to train: enable train_cfm and/or train_ar")


def run(cfg: FinetuneConfig, dataset_dir: Optional[str] = None,
        dry_run: bool = False) -> int:
    preflight(cfg, dataset_dir)
    cmd = build_command(cfg, dataset_dir)
    repo = str(Path(cfg.train.repo_dir).resolve())

    printable = " ".join(_quote(c) for c in cmd)
    log.info("cwd: %s", repo)
    log.info("run: %s", printable)
    if dry_run:
        print(printable)
        return 0

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(cmd, cwd=repo, env=env)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(timeout=30)
        return 130


def checkpoints(cfg: FinetuneConfig) -> List[Path]:
    """Checkpoints seed-vc has written for this run, newest last."""
    run_dir = Path(cfg.train.repo_dir) / "runs" / cfg.train.run_name
    if not run_dir.exists():
        return []
    return sorted(run_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)


def _quote(value: str) -> str:
    return f'"{value}"' if " " in value else value
