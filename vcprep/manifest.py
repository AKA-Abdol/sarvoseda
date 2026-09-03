"""JSONL manifest: the single source of truth for the pipeline.

Every utterance is one record. Nodes annotate records in place; nothing is
deleted. Because scores live in the manifest rather than being implied by which
folder a file sits in, retuning a threshold is a re-run of ``materialize``
(seconds) instead of a re-run of DNSMOS over 400 hours (days).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

#: stage names, in pipeline order
STAGES = ["fetch", "prefilter", "separate", "vad", "quality", "materialize"]

# verdicts
KEEP = "keep"
LOW_QUALITY = "low_quality"
REJECTED = "rejected"
PENDING = "pending"


@dataclass
class Record:
    """One utterance as it moves through the pipeline."""
    uid: str                       # "<source-slug>__<original-stem>"
    source: str = ""               # HF dataset slug this clip came from
    shard: int = -1
    source_name: str = ""          # original name inside the tar
    text: str = ""                 # transcript shipped with the dataset
    path: str = ""                 # current on-disk location (updated per stage)
    duration: float = 0.0
    sample_rate: int = 0

    status: str = PENDING          # keep | low_quality | rejected | pending
    stage: str = "fetch"           # last stage completed
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "Record":
        data = json.loads(line)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def reject(self, reason: str, status: str = REJECTED) -> None:
        self.status = status
        if reason not in self.reasons:
            self.reasons.append(reason)

    @property
    def alive(self) -> bool:
        """Still a candidate for the clean set."""
        return self.status in (PENDING, KEEP)


class Manifest:
    """Append-only JSONL with an in-memory index, rewritten atomically.

    Append-only during a run means a crash costs at most the records written
    since the last flush; ``compact()`` collapses duplicate uids, keeping the
    last write.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, Record] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            self.load()

    # ------------------------------------------------------------------ io
    def load(self) -> None:
        self._records.clear()
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = Record.from_json(line)
                except (json.JSONDecodeError, TypeError):
                    continue          # tolerate a torn final line after a crash
                self._records[rec.uid] = rec

    def flush(self) -> None:
        """Rewrite the manifest atomically from the in-memory index."""
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    for rec in self._records.values():
                        fh.write(rec.to_json() + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

    def compact(self) -> None:
        self.load()
        self.flush()

    # -------------------------------------------------------------- access
    def add(self, rec: Record) -> None:
        with self._lock:
            self._records[rec.uid] = rec

    def update(self, rec: Record) -> None:
        self.add(rec)

    def get(self, uid: str) -> Optional[Record]:
        return self._records.get(uid)

    def __contains__(self, uid: str) -> bool:
        return uid in self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Record]:
        return iter(list(self._records.values()))

    def records(self) -> List[Record]:
        return list(self._records.values())

    def pending_for(self, stage: str, shard: Optional[int] = None,
                    source: Optional[str] = None) -> List[Record]:
        """Records that are still alive and have not yet cleared ``stage``.

        This is what makes every node resumable: re-running a stage picks up
        exactly where it stopped, for one (source, shard) work unit at a time.
        """
        target = STAGES.index(stage)
        out = []
        for rec in self._records.values():
            if not rec.alive:
                continue
            if shard is not None and rec.shard != shard:
                continue
            if source is not None and rec.source != source:
                continue
            try:
                done = STAGES.index(rec.stage)
            except ValueError:
                done = -1
            if done < target:
                out.append(rec)
        return out

    def count_for_source(self, source: str) -> int:
        return sum(1 for r in self._records.values() if r.source == source)

    def by_status(self, status: str) -> List[Record]:
        return [r for r in self._records.values() if r.status == status]

    # --------------------------------------------------------------- stats
    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        stages: Dict[str, int] = {}
        reasons: Dict[str, int] = {}
        hours = 0.0
        for rec in self._records.values():
            counts[rec.status] = counts.get(rec.status, 0) + 1
            stages[rec.stage] = stages.get(rec.stage, 0) + 1
            for reason in rec.reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
            if rec.status == KEEP:
                hours += rec.duration
        per_source: Dict[str, Dict[str, Any]] = {}
        for rec in self._records.values():
            bucket = per_source.setdefault(
                rec.source or "?", {"total": 0, "keep": 0, "keep_hours": 0.0})
            bucket["total"] += 1
            if rec.status == KEEP:
                bucket["keep"] += 1
                bucket["keep_hours"] += rec.duration
        for bucket in per_source.values():
            bucket["keep_hours"] = round(bucket["keep_hours"] / 3600.0, 2)

        return {
            "total": len(self._records),
            "by_status": counts,
            "by_stage": stages,
            "by_source": per_source,
            "reject_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            "keep_hours": round(hours / 3600.0, 2),
        }


def unit_key(source: str, shard: Optional[int]) -> str:
    """Partition name for one (source, shard) work unit."""
    return f"shard_{shard:03d}" if shard is not None else "single"


class ManifestStore:
    """Per-work-unit manifest partitions.

    One JSONL per (source, shard) instead of one global file. That is what lets
    several work units run concurrently without contending on a single writer,
    and it keeps each partition small enough to rewrite atomically on every
    flush. Aggregate views (stats, calibration) read across partitions.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._open: Dict[str, Manifest] = {}
        self._known_uids: Optional[set] = None

    # ------------------------------------------------------------------
    def path_for(self, source: str, shard: Optional[int]) -> Path:
        return self.root / (source or "unknown") / f"{unit_key(source, shard)}.jsonl"

    def partition(self, source: str, shard: Optional[int]) -> Manifest:
        """The manifest for one work unit, cached per store."""
        path = self.path_for(source, shard)
        key = str(path)
        if key not in self._open:
            self._open[key] = Manifest(key)
        return self._open[key]

    def partitions(self) -> List[Path]:
        return sorted(self.root.glob("*/*.jsonl"))

    # ------------------------------------------------------------------
    def records(self) -> Iterator[Record]:
        """Stream every record across every partition."""
        for path in self.partitions():
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield Record.from_json(line)
                    except (json.JSONDecodeError, TypeError):
                        continue

    def known_uids(self) -> set:
        """Every uid already ingested, across partitions.

        Built once per process. Filenames are unique per shard in practice, but
        a global check is what actually guarantees a clip is never ingested
        twice when shard boundaries move or a source is re-added.
        """
        if self._known_uids is None:
            self._known_uids = {rec.uid for rec in self.records()}
        return self._known_uids

    def note_uid(self, uid: str) -> None:
        if self._known_uids is not None:
            self._known_uids.add(uid)

    def flush_all(self) -> None:
        for manifest in self._open.values():
            manifest.flush()

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Aggregate the same shape as :meth:`Manifest.summary`."""
        merged = Manifest.__new__(Manifest)
        merged._records = {r.uid: r for r in self.records()}
        merged._lock = threading.Lock()
        merged.path = self.root
        return merged.summary()

    def as_manifest(self) -> Manifest:
        """A read-only in-memory Manifest over every partition.

        Used by ``stats`` and ``calibrate``, which need the whole corpus at
        once; flushing this would collapse the partitions, so never do.
        """
        merged = Manifest.__new__(Manifest)
        merged._records = {r.uid: r for r in self.records()}
        merged._lock = threading.Lock()
        merged.path = self.root
        return merged
