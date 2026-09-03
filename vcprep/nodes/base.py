"""Common node plumbing."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from ..config import PipelineConfig
from ..manifest import Manifest, Record

log = logging.getLogger(__name__)


class Node(ABC):
    """One pipeline stage.

    Nodes read and write a manifest and are individually resumable: a node only
    touches records that are still ``alive`` and have not yet cleared its stage.

    Nodes are built once and reused across work units - loading a separation
    model or DNSMOS per shard would dominate the runtime - so the manifest and
    execution backend are *bound* per unit rather than captured at construction.
    """

    name: str = "node"

    def __init__(self, cfg: PipelineConfig, manifest: Optional[Manifest] = None):
        self.cfg = cfg
        self.manifest = manifest
        self._backend = None

    # -------------------------------------------------------------- binding
    def bind(self, manifest: Manifest, backend=None) -> "Node":
        self.manifest = manifest
        if backend is not None:
            self._backend = backend
        return self

    @property
    def backend(self):
        if self._backend is None:
            from ..backends import SerialBackend
            self._backend = SerialBackend(self.cfg)
        return self._backend

    @property
    def enabled(self) -> bool:
        section = getattr(self.cfg, self.name, None)
        return bool(getattr(section, "enabled", True)) if section else True

    # ------------------------------------------------------------------
    @abstractmethod
    def run(self, shard: Optional[int] = None,
            source: Optional[str] = None) -> dict:
        """Process pending records; return a small stats dict."""

    # -------------------------------------------------------------- helpers
    def pending(self, shard: Optional[int] = None,
                source: Optional[str] = None) -> List[Record]:
        return self.manifest.pending_for(self.name, shard=shard, source=source)

    def advance(self, rec: Record) -> None:
        """Mark this stage complete for ``rec``."""
        rec.stage = self.name
        self.manifest.update(rec)

    @staticmethod
    def unit_dir(root, source: Optional[str], shard: Optional[int]) -> Path:
        """Per-(source, shard) working directory, so stages never mix repos."""
        out = Path(root) / (source or "all")
        out = out / (f"shard_{shard:03d}" if shard is not None else "single")
        out.mkdir(parents=True, exist_ok=True)
        return out
