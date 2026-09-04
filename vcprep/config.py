"""Typed configuration for the preprocessing pipeline.

Everything is overridable from YAML (``--config``) and then from CLI flags,
in that order of increasing precedence.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_REPO = "MohammadGholizadeh/filimo-farsi-raw"

# Shards 010 and 015 are absent from the repo; the fetcher tolerates gaps but
# listing the real set keeps progress reporting honest.
DEFAULT_SHARDS = [i for i in range(1, 34) if i not in (10, 15)]


@dataclass
class DatasetSource:
    """One Hugging Face dataset to ingest.

    Several sources can be queued in a single run; every utterance carries its
    source slug, and outputs are foldered and named by it, so a mixed corpus
    stays traceable back to the repo each clip came from.
    """
    repo_id: str = DEFAULT_REPO
    #: output slug. Defaults to the repo basename (``filimo-farsi-raw``);
    #: set explicitly when two repos share a basename.
    name: str = ""
    revision: str = "main"
    enabled: bool = True

    #: auto | tar | parquet | files
    layout: str = "auto"

    # --- tar layout (filimo-farsi-raw and friends) ---
    tar_glob: str = "data/*.tar"
    #: explicit shard numbers, or empty to take every tar the repo actually has
    shards: List[int] = field(default_factory=list)

    # --- parquet layout (standard HF audio datasets) ---
    parquet_glob: str = "**/*.parquet"
    audio_column: str = "audio"
    text_column: str = ""          # auto-detected from common names

    # --- files layout (loose audio in the repo) ---
    file_globs: List[str] = field(default_factory=lambda: [
        "**/*.mp3", "**/*.wav", "**/*.flac", "**/*.ogg", "**/*.opus", "**/*.m4a",
    ])

    # --- optional transcript sidecar ---
    metadata_file: str = ""
    metadata_delimiter: str = "\t"
    metadata_key: str = "file_name"
    metadata_text: str = "sentence"

    #: caps for smoke tests / disk budgeting (0 = unlimited)
    limit: int = 0
    limit_per_shard: int = 0

    hf_token: Optional[str] = None

    @property
    def slug(self) -> str:
        """Filesystem-safe identity used in uids, folders and filenames."""
        raw = self.name or self.repo_id.split("/")[-1]
        return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in raw)

    @classmethod
    def from_spec(cls, spec: str) -> "DatasetSource":
        """Parse a CLI ``--repo`` value: ``owner/name[@revision][:name=slug]``."""
        rest, _, alias = spec.partition("#")
        repo, _, revision = rest.partition("@")
        src = cls(repo_id=repo.strip())
        if revision.strip():
            src.revision = revision.strip()
        if alias.strip():
            src.name = alias.strip()
        return src


@dataclass
class FetchConfig:
    """Global ingestion behaviour, shared across every source."""
    enabled: bool = True
    #: stream tars over HTTP instead of landing the archive on disk
    stream: bool = True
    #: delete each raw clip once it has been separated
    delete_after_separate: bool = True

    #: How to pull tar shards.
    #:   "stream"    one connection, unpacked on the fly, no tar on disk
    #:   "parallel"  many ranged connections into a temp file, then unpacked
    #: A single TCP stream on a long-haul route is capped by the
    #: bandwidth-delay product well before the link itself saturates, so
    #: parallel is usually several times faster. Measure yours with
    #: `vcprep netcheck`. Costs one shard of temporary disk (~1-1.5 GB),
    #: reclaimed as soon as the shard is unpacked.
    download_mode: str = "parallel"
    #: simultaneous connections when download_mode is "parallel"
    connections: int = 8
    #: bytes per ranged request; more requests than connections balances load
    chunk_bytes: int = 16 * 1024 * 1024
    #: retries per chunk before the shard is abandoned
    chunk_retries: int = 3


@dataclass
class PrefilterConfig:
    """Cheap gate that runs on *raw* audio, before the GPU-bound separator."""
    enabled: bool = True
    #: reject if the whole file is quieter than this (dBFS)
    min_dbfs: float = -55.0
    #: reject clips shorter than this many seconds (seed-vc wants >= 1s)
    min_duration: float = 1.0
    #: reject clips longer than this (seed-vc v2 wants <= 30s)
    max_duration: float = 30.0
    #: run Silero VAD in the prefilter too. Deliberately lenient here - this
    #: stage only exists to avoid paying GPU separation cost on dead clips.
    use_vad: bool = True
    min_speech_seconds: float = 0.4
    vad_threshold: float = 0.30


@dataclass
class SeparateConfig:
    """Vocal isolation.

    Defaults to Mel-Band RoFormer, which scores 12.60 vocal SDR against Kim
    Vocal 2's 10.18 on the same benchmark. Any model in audio-separator's
    registry works; unknown-but-present local files work too.
    """
    enabled: bool = True
    #: registry name or a bare filename found in ``model_dir``
    model_name: str = "vocals_mel_band_roformer.ckpt"
    model_dir: str = "models/separator"
    #: convenience: an absolute path, split into model_dir + model_name
    model_path: str = ""

    output_format: str = "WAV"
    sample_rate: int = 44100
    normalization_threshold: float = 0.9
    use_autocast: bool = False
    keep_instrumental: bool = False

    # --- architecture parameters (only the matching one is used) ---
    mdx_params: Dict[str, Any] = field(default_factory=lambda: {
        "segment_size": 256, "overlap": 0.25, "batch_size": 4,
        "hop_length": 1024, "enable_denoise": False,
    })
    mdxc_params: Dict[str, Any] = field(default_factory=lambda: {
        "segment_size": 256, "override_model_segment_size": False,
        "batch_size": 4, "overlap": 8, "pitch_shift": 0,
    })

    # --- batching -----------------------------------------------------
    # These models run a fixed-size inference window (Kim Vocal 2: 256 frames
    # x 1024 hop = 5.94 s at 44.1 kHz), so a 2-second clip costs as much as a
    # 6-second one. With utterances averaging ~2 s, separating them one at a
    # time wastes most of the GPU. Packing many clips into one long input and
    # slicing the result apart afterwards measured ~6x faster.
    batch_clips: bool = True
    #: seconds of audio per packed input
    batch_seconds: float = 300.0
    #: silence inserted between packed clips, so neighbours cannot bleed
    #: across a boundary inside the model's receptive field
    batch_guard_seconds: float = 0.5
    #: peak each clip is scaled to before packing, with the gain restored
    #: afterwards. Without this the separator's global normalisation would make
    #: a clip's result depend on how loud its batch-mates happened to be.
    batch_peak: float = 0.7

    def resolve(self) -> "SeparateConfig":
        if self.model_path:
            path = Path(os.path.expanduser(self.model_path))
            self.model_name = path.name
            self.model_dir = str(path.parent)
        return self


@dataclass
class VadConfig:
    """Silence / no-speech detection on the *separated* vocals."""
    enabled: bool = True
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 300
    speech_pad_ms: int = 120
    #: hard rejects
    min_speech_seconds: float = 1.0
    #: speech seconds / total seconds; separation residue scores very low here
    min_speech_ratio: float = 0.45
    #: absolute level gate applied after separation
    min_dbfs: float = -50.0
    #: trim to [first speech - pad, last speech + pad]
    trim: bool = True
    trim_pad_ms: int = 150
    #: drop clips whose trimmed length leaves seed-vc's window
    min_trimmed_seconds: float = 1.0
    max_trimmed_seconds: float = 30.0


@dataclass
class QualityConfig:
    """Non-intrusive quality scoring."""
    enabled: bool = True

    # --- which scorers to run ---
    use_dnsmos: bool = True
    use_heuristics: bool = True
    use_squim: bool = False
    use_nisqa: bool = False

    # --- model locations ---
    dnsmos_dir: str = "models/dnsmos"      # holds sig_bak_ovr.onnx
    nisqa_repo: str = "third_party/NISQA"  # cloned by scripts/setup_nisqa.sh
    nisqa_weights: str = "third_party/NISQA/weights/nisqa.tar"

    # --- thresholds (run `vcprep calibrate` before trusting these) ---
    min_ovrl: float = 2.60      # DNSMOS P.835 overall
    min_sig: float = 3.00       # speech signal quality
    min_bak: float = 3.20       # background intrusiveness -> grades UVR itself
    min_nisqa_dis: float = 3.00  # discontinuity: catches MDX warbling
    #: streaming rips are often lowpassed; a 22.05k seed-vc model needs headroom
    min_bandwidth_hz: float = 7000.0
    max_clipping_ratio: float = 0.001
    #: Where DNSMOS (and SQUIM) run.
    #:   "cuda"  always the GPU, whatever the worker count
    #:   "cpu"   always the CPU, spread across the pool
    #:   "auto"  GPU if available AND workers <= max_gpu_workers, else CPU
    #:
    #: Neither is universally right. A typical clip is a single 9 s window at
    #: batch size 1, so DNSMOS is latency-bound and gains little from a GPU -
    #: with many cores, CPU workers win outright. But every worker builds its
    #: own CUDA context, so on a box with few cores the CPU pool is too small
    #: to keep up and the GPU is the better home. Two cores is squarely in
    #: that second case.
    device: str = "auto"        # auto | cpu | cuda
    #: under "auto", the worker count above which DNSMOS is pinned to CPU
    #: rather than opening one CUDA context per worker
    max_gpu_workers: int = 4


@dataclass
class MaterializeConfig:
    """Turn manifest verdicts into folders on disk."""
    enabled: bool = True
    clean_dir: str = "clean"
    low_quality_dir: str = "low_quality"
    rejected_dir: str = "rejected"
    #: 'copy' keeps stage outputs, 'move' reclaims disk as it goes
    mode: str = "move"
    #: final training format
    sample_rate: int = 22050     # seed-vc v2
    output_format: str = "FLAC"
    mono: bool = True
    #: write rejected audio at all, or just record the verdict in the manifest
    keep_rejected_audio: bool = False


@dataclass
class PathsConfig:
    work_dir: str = "work"
    out_dir: str = "out"
    #: directory of per-work-unit manifest partitions
    #: (<manifest_dir>/<source>/shard_NNN.jsonl)
    manifest_dir: str = ""

    def resolve(self) -> "PathsConfig":
        if not self.manifest_dir:
            self.manifest_dir = str(Path(self.work_dir) / "manifest")
        return self


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    sources: List[DatasetSource] = field(
        default_factory=lambda: [DatasetSource(repo_id=DEFAULT_REPO,
                                              shards=list(DEFAULT_SHARDS))])
    fetch: FetchConfig = field(default_factory=FetchConfig)
    prefilter: PrefilterConfig = field(default_factory=PrefilterConfig)
    separate: SeparateConfig = field(default_factory=SeparateConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    materialize: MaterializeConfig = field(default_factory=MaterializeConfig)

    #: execution backend for the CPU stages: serial | process | ray.
    #: 'process' is the right default on a single server; 'ray' only pays for
    #: itself across several machines.
    backend: str = "process"
    #: worker count for the backend; 0 = one per CPU core
    num_workers: int = 0
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str] = None) -> "PipelineConfig":
        cfg = cls()
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            cfg = _merge(cfg, data)
        cfg.paths.resolve()
        cfg.separate.resolve()
        return cfg

    # ------------------------------------------------------------------
    def active_sources(self) -> List[DatasetSource]:
        return [s for s in self.sources if s.enabled]

    def source_by_slug(self, slug: str) -> Optional[DatasetSource]:
        for src in self.sources:
            if src.slug == slug:
                return src
        return None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)


def _merge(obj: Any, data: Dict[str, Any]) -> Any:
    """Recursively overlay a plain dict onto a dataclass instance."""
    if not is_dataclass(obj):
        return data
    known = {f.name: f for f in fields(obj)}
    for key, value in (data or {}).items():
        if key not in known:
            raise KeyError(f"unknown config key: {key!r} (in {type(obj).__name__})")
        if key == "sources":
            setattr(obj, key, [_merge(DatasetSource(), item) for item in value])
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(obj, key, _merge(current, value))
        else:
            setattr(obj, key, value)
    return obj
