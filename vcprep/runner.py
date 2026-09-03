"""Pipeline orchestration.

Work is driven one (source, shard) unit at a time and carried all the way to
``materialize`` before the next unit starts. That ordering keeps the disk
budget flat: the machine holds one shard's audio at a time, not the whole
corpus, and each stage deletes its input once its output is written.

Parallelism has two independent axes, because the stages have different
bottlenecks:

* the CPU stages (prefilter, VAD, quality) fan out across cores through a
  pluggable :mod:`~vcprep.backends` backend;
* separation keeps the GPU to itself and gets its speedup from packing many
  short clips into one long input instead - running several copies of it would
  multiply VRAM, not throughput.

Each work unit writes its own manifest partition, so units never contend on a
single writer.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from .backends import make_backend
from .config import PipelineConfig
from .manifest import ManifestStore
from .nodes.fetch import FetchNode, WorkUnit
from .nodes.materialize import MaterializeNode
from .nodes.prefilter import PrefilterNode
from .nodes.quality import QualityNode
from .nodes.separate import SeparateNode
from .nodes.vad import VadNode

log = logging.getLogger(__name__)

STAGE_ORDER = ["fetch", "prefilter", "separate", "vad", "quality", "materialize"]

NODE_TYPES = {
    "fetch": FetchNode,
    "prefilter": PrefilterNode,
    "separate": SeparateNode,
    "vad": VadNode,
    "quality": QualityNode,
    "materialize": MaterializeNode,
}

#: stages that go through the execution backend
PARALLEL_STAGES = {"prefilter", "vad", "quality"}


class Pipeline:
    def __init__(self, cfg: PipelineConfig, stages: Optional[List[str]] = None,
                 backend: Optional[str] = None, num_workers: int = 0,
                 ray_address: Optional[str] = None):
        self.cfg = cfg
        self.store = ManifestStore(cfg.paths.manifest_dir)
        self.stages = self._resolve_stages(stages)
        self.backend = make_backend(backend or cfg.backend, cfg,
                                    num_workers or cfg.num_workers,
                                    address=ray_address)
        # Built once and reused across units; the manifest is bound per unit.
        self.nodes = {name: NODE_TYPES[name](cfg) for name in STAGE_ORDER}
        log.info("active stages: %s | backend: %s x%d",
                 ", ".join(self.stages), self.backend.name,
                 self.backend.num_workers)

    def _resolve_stages(self, stages: Optional[List[str]]) -> List[str]:
        if not stages:
            return [s for s in STAGE_ORDER if self._config_enabled(s)]
        unknown = set(stages) - set(STAGE_ORDER)
        if unknown:
            raise ValueError(f"unknown stage(s): {sorted(unknown)}")
        return [s for s in STAGE_ORDER if s in stages]

    def _config_enabled(self, stage: str) -> bool:
        section = getattr(self.cfg, stage, None)
        return bool(getattr(section, "enabled", True)) if section else True

    # ------------------------------------------------------------------
    def plan(self, only_sources: Optional[List[str]] = None) -> List[WorkUnit]:
        fetch: FetchNode = self.nodes["fetch"]           # type: ignore[assignment]
        sources = self.cfg.active_sources()
        if only_sources:
            wanted = set(only_sources)
            sources = [s for s in sources
                       if s.slug in wanted or s.repo_id in wanted]
            if not sources:
                raise KeyError(f"no configured source matches {only_sources}")
        return fetch.plan(sources)

    # ------------------------------------------------------------------
    def run(self, only_sources: Optional[List[str]] = None,
            only_shards: Optional[List[int]] = None,
            keep_intermediates: bool = False) -> Dict[str, dict]:
        units = self.plan(only_sources)
        if only_shards is not None:
            units = [u for u in units if u.shard in set(only_shards)]
        if not units:
            log.warning("nothing to do")
            return {}

        log.info("queued %d work unit(s): %s", len(units),
                 ", ".join(u.label for u in units[:8])
                 + (" ..." if len(units) > 8 else ""))

        totals: Dict[str, dict] = {}
        try:
            for index, unit in enumerate(units, start=1):
                started = time.time()
                log.info("=== [%d/%d] %s ===", index, len(units), unit.label)
                try:
                    stats = self.run_unit(unit, keep_intermediates)
                except KeyboardInterrupt:
                    log.warning("interrupted - manifests flushed, re-run to resume")
                    raise
                except Exception as exc:
                    # One bad shard must not take the queue down; the manifest
                    # records how far it got and a re-run resumes it.
                    log.exception("work unit %s failed: %s", unit.label, exc)
                    continue
                totals[unit.label] = stats
                log.info("--- %s done in %.1fs: %s", unit.label,
                         time.time() - started, _brief(stats))
        finally:
            self.store.flush_all()
            self.backend.close()
        return totals

    # ------------------------------------------------------------------
    def run_unit(self, unit: WorkUnit, keep_intermediates: bool = False) -> dict:
        slug, shard = unit.source.slug, unit.shard
        manifest = self.store.partition(slug, shard)
        stats: Dict[str, dict] = {}

        for stage in self.stages:
            node = self.nodes[stage]
            node.bind(manifest,
                      self.backend if stage in PARALLEL_STAGES else None)
            if stage in ("fetch", "materialize"):
                node.store = self.store            # type: ignore[attr-defined]
            if stage == "fetch":
                added, skipped = node.fetch_unit(unit)   # type: ignore[attr-defined]
                stats["fetch"] = {"added": added, "skipped": skipped}
            else:
                stats[stage] = node.run(shard=shard, source=slug)

        manifest.flush()
        if not keep_intermediates:
            self._cleanup(slug, shard)
        return stats

    # ------------------------------------------------------------------
    def _cleanup(self, slug: str, shard: Optional[int]) -> None:
        """Drop this unit's scratch directories once it has been materialised."""
        work = Path(self.cfg.paths.work_dir)
        sub = f"shard_{shard:03d}" if shard is not None else "single"
        for stage_dir in ("raw", "vocals", "trimmed"):
            for candidate in (work / stage_dir / slug / sub,
                              work / stage_dir / slug / "files",
                              work / stage_dir / slug / "parquet",
                              work / stage_dir / slug / "single"):
                if candidate.exists():
                    shutil.rmtree(candidate, ignore_errors=True)


def _brief(stats: Dict[str, dict]) -> str:
    parts = []
    for stage, values in stats.items():
        if not isinstance(values, dict):
            continue
        inner = " ".join(f"{k}={v}" for k, v in values.items()
                         if isinstance(v, (int, float)))
        if inner:
            parts.append(f"{stage}[{inner}]")
    return " ".join(parts)
