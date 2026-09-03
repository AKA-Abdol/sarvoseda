"""Configuration for seed-vc v2 fine-tuning.

Deliberately separate from :mod:`vcprep`. The two halves share nothing but the
on-disk contract: preprocessing produces ``out/clean/<source>/*.flac`` plus a
``metadata.csv``, and this half consumes that directory.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    """How to turn pipeline output into a seed-vc dataset directory."""
    #: pipeline output root, or a clean/ directory directly
    clean_dir: str = "out/clean"
    #: where the prepared training set is assembled
    dataset_dir: str = "seedvc_data"
    #: optionally include the demoted clips too (usually a bad idea)
    include_low_quality: bool = False
    low_quality_dir: str = "out/low_quality"
    #: restrict to these source slugs (empty = all)
    sources: List[str] = field(default_factory=list)

    # seed-vc trains on 1-30 s utterances
    min_duration: float = 1.0
    max_duration: float = 30.0
    sample_rate: int = 22050

    #: link | copy | symlink - 'link' (hardlink) costs no extra disk
    materialize: str = "link"
    #: cap the training set (0 = all); useful for a first convergence check
    max_files: int = 0
    #: held-out fraction, written next to the training set for eval
    val_fraction: float = 0.01
    seed: int = 1234

    #: extra quality floor applied on top of the pipeline's own thresholds,
    #: read from metadata.csv. Empty = take whatever the pipeline kept.
    min_dnsmos_ovrl: float = 0.0
    min_dnsmos_bak: float = 0.0


@dataclass
class TrainConfig:
    """Arguments forwarded to seed-vc's ``train_v2.py``."""
    #: clone of https://github.com/Plachtaa/seed-vc
    repo_dir: str = "third_party/seed-vc"
    config: str = "configs/v2/vc_wrapper.yaml"
    run_name: str = "sarvoseda_fa"

    batch_size: int = 4
    max_steps: int = 20000
    max_epochs: int = 1000
    save_every: int = 1000
    num_workers: int = 4

    #: V2 splits into a CFM (acoustic) and an AR (prosody/timbre) stage.
    #: Fine-tuning the CFM is the one that matters for accent and voice
    #: quality on a new language, and it is far cheaper; the AR stage needs
    #: considerably more data before it stops overfitting.
    train_cfm: bool = True
    train_ar: bool = False

    pretrained_cfm_ckpt: Optional[str] = None
    pretrained_ar_ckpt: Optional[str] = None

    #: accelerate launch options
    num_processes: int = 1
    mixed_precision: str = "fp16"     # no | fp16 | bf16
    main_process_port: int = 29500
    extra_args: List[str] = field(default_factory=list)


@dataclass
class FinetuneConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: Optional[str] = None) -> "FinetuneConfig":
        cfg = cls()
        if path:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            cfg = _merge(cfg, data)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)


def _merge(obj: Any, data: Dict[str, Any]) -> Any:
    if not is_dataclass(obj):
        return data
    known = {f.name for f in fields(obj)}
    for key, value in (data or {}).items():
        if key not in known:
            raise KeyError(f"unknown config key: {key!r} (in {type(obj).__name__})")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(obj, key, _merge(current, value))
        else:
            setattr(obj, key, value)
    return obj
