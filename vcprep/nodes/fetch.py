"""Stage 1 - pull utterances out of one or more Hugging Face datasets.

Several datasets can be queued in a single run. Every utterance is tagged with
its source slug and gets a uid of ``<slug>__<original-stem>``, so a mixed
corpus never collides and every output file names the repo it came from.

Three repo layouts are handled, auto-detected per source:

``tar``      sharded tar archives (``data/unvalidated_001.tar``, ...). Streamed
             over HTTP and unpacked on the fly, so the ~1 GB archive is never
             written to disk - the pipeline only ever holds one shard's audio.
``parquet``  the standard HF audio-dataset layout, where the audio column holds
             ``{bytes, path}`` structs. Read row-group by row-group.
``files``    loose audio files committed to the repo.
"""
from __future__ import annotations

import csv
import fnmatch
import io
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

from ..config import DatasetSource, PipelineConfig
from ..manifest import Manifest, Record
from .base import Node

log = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}

#: Below this many wanted clips, streaming beats downloading a whole shard.
SMALL_BUDGET = 500
TEXT_COLUMN_CANDIDATES = ("sentence", "text", "transcription", "transcript",
                          "normalized_text")


class WorkUnit:
    """One (source, shard) pair - the granularity the runner streams at."""

    __slots__ = ("source", "shard", "label")

    def __init__(self, source: DatasetSource, shard: Optional[int], label: str):
        self.source = source
        self.shard = shard
        self.label = label

    def __repr__(self) -> str:
        return f"<WorkUnit {self.label}>"


class FetchNode(Node):
    name = "fetch"

    def __init__(self, cfg: PipelineConfig, manifest: Optional[Manifest] = None):
        super().__init__(cfg, manifest)
        self.raw_root = Path(cfg.paths.work_dir) / "raw"
        #: set by the runner; gives a corpus-wide uid check rather than a
        #: per-partition one, so a clip is never ingested twice when shard
        #: boundaries move or a source is re-added.
        self.store = None
        self._metadata_cache: Dict[str, Dict[str, str]] = {}
        self._files_cache: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # repo introspection
    # ------------------------------------------------------------------
    def repo_files(self, src: DatasetSource) -> List[str]:
        key = f"{src.repo_id}@{src.revision}"
        if key not in self._files_cache:
            from huggingface_hub import HfApi
            api = HfApi()
            self._files_cache[key] = api.list_repo_files(
                repo_id=src.repo_id, repo_type="dataset",
                revision=src.revision, token=src.hf_token,
            )
        return self._files_cache[key]

    def detect_layout(self, src: DatasetSource) -> str:
        if src.layout != "auto":
            return src.layout
        files = self.repo_files(src)
        if any(fnmatch.fnmatch(f, src.tar_glob) for f in files):
            return "tar"
        if any(f.endswith(".parquet") for f in files):
            return "parquet"
        if any(Path(f).suffix.lower() in AUDIO_EXTS for f in files):
            return "files"
        raise RuntimeError(
            f"cannot determine layout for {src.repo_id}; set layout explicitly"
        )

    def plan(self, sources: Optional[List[DatasetSource]] = None) -> List[WorkUnit]:
        """Expand the configured sources into an ordered queue of work units."""
        units: List[WorkUnit] = []
        for src in (sources if sources is not None else self.cfg.active_sources()):
            layout = self.detect_layout(src)
            if layout == "tar":
                for shard in self._shard_numbers(src):
                    units.append(WorkUnit(src, shard, f"{src.slug}/shard_{shard:03d}"))
            else:
                units.append(WorkUnit(src, None, f"{src.slug}/{layout}"))
        return units

    def _shard_numbers(self, src: DatasetSource) -> List[int]:
        """Shard ids actually present in the repo, intersected with config.

        filimo-farsi-raw is missing shards 010 and 015; discovering rather than
        assuming keeps progress reporting honest.
        """
        present = []
        for name in self.repo_files(src):
            if not fnmatch.fnmatch(name, src.tar_glob):
                continue
            digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
            if digits:
                present.append(int(digits))
        present = sorted(set(present))
        if src.shards:
            wanted = set(src.shards)
            missing = wanted - set(present)
            if missing:
                log.warning("%s: requested shards not in repo: %s",
                            src.slug, sorted(missing))
            return [s for s in present if s in wanted]
        return present

    # ------------------------------------------------------------------
    # transcripts
    # ------------------------------------------------------------------
    def metadata(self, src: DatasetSource) -> Dict[str, str]:
        if not src.metadata_file:
            return {}
        if src.slug in self._metadata_cache:
            return self._metadata_cache[src.slug]

        from huggingface_hub import hf_hub_download
        log.info("%s: downloading transcripts (%s)", src.slug, src.metadata_file)
        path = hf_hub_download(repo_id=src.repo_id, filename=src.metadata_file,
                               repo_type="dataset", revision=src.revision,
                               token=src.hf_token)
        mapping: Dict[str, str] = {}
        # filimo's "unvalidated.csv" is in fact TAB separated and unquoted -
        # the dataset's own loader reads it with QUOTE_NONE.
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=src.metadata_delimiter,
                                    quoting=csv.QUOTE_NONE)
            for row in reader:
                key = (row.get(src.metadata_key) or "").strip()
                if not key:
                    continue
                mapping[Path(key).stem] = (row.get(src.metadata_text) or "").strip()
        log.info("%s: %d transcripts", src.slug, len(mapping))
        self._metadata_cache[src.slug] = mapping
        return mapping

    # ------------------------------------------------------------------
    def run(self, shard: Optional[int] = None,
            source: Optional[str] = None) -> dict:
        """Fetch everything (or one source/shard). The runner normally calls
        :meth:`fetch_unit` per work unit instead, to bound disk use."""
        stats = {"units": 0, "added": 0, "skipped": 0}
        sources = None
        if source:
            found = self.cfg.source_by_slug(source)
            if not found:
                raise KeyError(f"no configured source with slug {source!r}")
            sources = [found]
        for unit in self.plan(sources):
            if shard is not None and unit.shard != shard:
                continue
            added, skipped = self.fetch_unit(unit)
            stats["units"] += 1
            stats["added"] += added
            stats["skipped"] += skipped
        return stats

    # ------------------------------------------------------------------
    def fetch_unit(self, unit: WorkUnit) -> Tuple[int, int]:
        """Materialise one work unit's audio into ``work/raw/<slug>/...``."""
        src = unit.source
        layout = self.detect_layout(src)
        subdir = f"shard_{unit.shard:03d}" if unit.shard is not None else layout
        dest = self.raw_root / src.slug / subdir
        dest.mkdir(parents=True, exist_ok=True)

        transcripts = self.metadata(src)
        added = skipped = 0

        # Budgets count records already in the manifest, not just what this
        # invocation adds - otherwise every resume would append another
        # limit_per_shard clips on top of the ones already fetched.
        budget = 0
        if src.limit_per_shard:
            have_here = sum(1 for r in self._all_records()
                            if r.source == src.slug and r.shard == unit.shard)
            budget = max(0, src.limit_per_shard - have_here)
            if budget == 0:
                log.info("%s: limit_per_shard %d already satisfied",
                         unit.label, src.limit_per_shard)
                return 0, 0
        if src.limit:
            have_total = sum(1 for r in self._all_records()
                             if r.source == src.slug)
            remaining = max(0, src.limit - have_total)
            if remaining == 0:
                log.info("%s: limit %d already reached", src.slug, src.limit)
                return 0, 0
            budget = min(budget, remaining) if budget else remaining

        if layout == "tar":
            stream = self._iter_tar(src, unit.shard, budget=budget)
        elif layout == "parquet":
            stream = self._iter_parquet(src)
        else:
            stream = self._iter_files(src)

        for member_name, payload, inline_text in stream:
            stem = Path(member_name).name
            if Path(stem).suffix.lower() not in AUDIO_EXTS:
                continue
            base = Path(stem).stem
            uid = f"{src.slug}__{base}"
            if self._seen(uid):
                skipped += 1
                continue

            out_path = dest / f"{uid}{Path(stem).suffix.lower()}"
            out_path.write_bytes(payload)

            self.manifest.add(Record(
                uid=uid,
                source=src.slug,
                shard=unit.shard if unit.shard is not None else -1,
                source_name=stem,
                text=inline_text or transcripts.get(base, ""),
                path=str(out_path),
                stage="fetch",
            ))
            if self.store is not None:
                self.store.note_uid(uid)
            added += 1
            if budget and added >= budget:
                log.info("%s: hit limit (%d)", unit.label, budget)
                break

        self.manifest.flush()
        log.info("%s: +%d utterances (%d already known)", unit.label, added, skipped)
        return added, skipped

    def _all_records(self):
        return self.store.records() if self.store is not None \
            else self.manifest.records()

    def _seen(self, uid: str) -> bool:
        if self.store is not None:
            return uid in self.store.known_uids()
        return uid in self.manifest

    # ------------------------------------------------------------------
    # layout readers -> (name, bytes, inline_text)
    # ------------------------------------------------------------------
    def _tar_filename(self, src: DatasetSource, shard: Optional[int]) -> str:
        """Repo path of a shard's tar, resolved from the actual file listing.

        Shard ids are discovered from filenames rather than templated, so the
        reverse lookup has to go through the listing too - the repo's naming is
        not ours to assume.
        """
        for name in self.repo_files(src):
            if not fnmatch.fnmatch(name, src.tar_glob):
                continue
            digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
            if digits and int(digits) == shard:
                return name
        raise FileNotFoundError(f"no tar for shard {shard} in {src.repo_id}")

    def _iter_tar(self, src: DatasetSource, shard: Optional[int],
                  budget: int = 0) -> Iterator[Tuple[str, bytes, str]]:
        from huggingface_hub import hf_hub_url

        filename = self._tar_filename(src, shard)
        url = hf_hub_url(repo_id=src.repo_id, filename=filename,
                         repo_type="dataset", revision=src.revision)
        headers = {}
        token = src.hf_token or os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        mode = self.cfg.fetch.download_mode
        # Parallel mode has to land the whole ~1 GB tar before it can unpack
        # anything. When only a handful of clips are wanted - a smoke test -
        # streaming stops after a few megabytes instead, which is far cheaper.
        if mode == "parallel" and 0 < budget <= SMALL_BUDGET:
            log.info("%s: only %d clip(s) wanted - streaming instead of "
                     "downloading the whole shard", filename, budget)
            mode = "stream"
        if mode == "parallel":
            yield from self._iter_tar_parallel(src, filename, url, headers)
        else:
            yield from self._iter_tar_stream(src, filename, url, headers)

    # ------------------------------------------------------------------
    def _iter_tar_stream(self, src, filename, url, headers
                         ) -> Iterator[Tuple[str, bytes, str]]:
        """One connection, unpacked on the fly. No tar ever touches disk."""
        import requests

        with requests.get(url, stream=True, headers=headers, timeout=120) as resp:
            if resp.status_code == 404:
                raise FileNotFoundError(filename)
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length") or 0)
            reader = _ProgressReader(resp.raw, total, desc=f"{src.slug} {filename}")
            # 'r|*' = streaming mode: sequential, no seeking, constant memory.
            with tarfile.open(fileobj=reader, mode="r|*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    handle = tar.extractfile(member)
                    if handle is None:
                        continue
                    yield member.name, handle.read(), ""

    # ------------------------------------------------------------------
    def _iter_tar_parallel(self, src, filename, url, headers
                           ) -> Iterator[Tuple[str, bytes, str]]:
        """Many ranged connections into a temp file, then unpack from disk.

        A single TCP stream is capped by the bandwidth-delay product on a
        long-haul route, so this is usually several times faster than
        streaming. It costs one shard of temporary disk, deleted as soon as
        the shard is unpacked. Falls back to streaming if the CDN will not
        serve ranges.
        """
        fetch = self.cfg.fetch
        tmp_dir = Path(self.cfg.paths.work_dir) / "_downloads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dest = tmp_dir / f"{src.slug}_{Path(filename).name}"

        try:
            _download_ranged(url, headers, dest,
                             connections=fetch.connections,
                             chunk_bytes=fetch.chunk_bytes,
                             retries=fetch.chunk_retries,
                             desc=f"{src.slug} {Path(filename).name}")
        except FileNotFoundError:
            raise
        except _RangesUnsupported:
            log.info("%s: server will not serve ranges - streaming instead",
                     filename)
            _unlink(dest)
            yield from self._iter_tar_stream(src, filename, url, headers)
            return

        try:
            with tarfile.open(dest, mode="r:*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    handle = tar.extractfile(member)
                    if handle is None:
                        continue
                    yield member.name, handle.read(), ""
        finally:
            # The tar is pure scratch - the audio has been written out by now,
            # and holding 1 GB per shard is exactly what we are avoiding.
            _unlink(dest)

    def _iter_parquet(self, src: DatasetSource) -> Iterator[Tuple[str, bytes, str]]:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        shards = [f for f in self.repo_files(src)
                  if fnmatch.fnmatch(f, src.parquet_glob) or f.endswith(".parquet")]
        if not shards:
            raise FileNotFoundError(f"{src.repo_id}: no parquet files matched")

        for rel in tqdm(sorted(shards), desc=f"{src.slug} parquet", leave=False):
            local = hf_hub_download(repo_id=src.repo_id, filename=rel,
                                    repo_type="dataset", revision=src.revision,
                                    token=src.hf_token)
            parquet = pq.ParquetFile(local)
            text_col = src.text_column or _find_text_column(parquet.schema_arrow.names)
            for batch in parquet.iter_batches(batch_size=64):
                rows = batch.to_pylist()
                for i, row in enumerate(rows):
                    audio = row.get(src.audio_column)
                    if not isinstance(audio, dict) or not audio.get("bytes"):
                        continue
                    name = audio.get("path") or f"{Path(rel).stem}_{i}.wav"
                    yield name, audio["bytes"], str(row.get(text_col, "") or "")
            # Reclaim the parquet immediately - these are often >500 MB.
            if src.limit == 0:
                _unlink(local)

    def _iter_files(self, src: DatasetSource) -> Iterator[Tuple[str, bytes, str]]:
        from huggingface_hub import hf_hub_download

        names = [f for f in self.repo_files(src)
                 if any(fnmatch.fnmatch(f, g) for g in src.file_globs)]
        for rel in tqdm(sorted(names), desc=f"{src.slug} files", leave=False):
            local = hf_hub_download(repo_id=src.repo_id, filename=rel,
                                    repo_type="dataset", revision=src.revision,
                                    token=src.hf_token)
            yield rel, Path(local).read_bytes(), ""


def _find_text_column(columns) -> str:
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return ""


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


class _ProgressReader(io.RawIOBase):
    """Wraps a stream so tarfile's reads drive a tqdm bar."""

    def __init__(self, stream, total: int, desc: str = ""):
        self._stream = stream
        self._bar = tqdm(total=total or None, unit="B", unit_scale=True,
                         unit_divisor=1024, desc=desc, leave=False)

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        chunk = self._stream.read(size)
        if chunk:
            self._bar.update(len(chunk))
        return chunk

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        self._bar.close()
        super().close()


class _RangesUnsupported(RuntimeError):
    """The server would not honour a Range request."""


def _download_ranged(url: str, headers: dict, dest: Path, connections: int = 8,
                     chunk_bytes: int = 16 * 1024 * 1024, retries: int = 3,
                     desc: str = "") -> Path:
    """Download ``url`` to ``dest`` using several simultaneous byte ranges.

    Chunks are written with ``os.pwrite`` at their own offsets, which is safe
    from multiple threads and avoids reassembling parts afterwards.
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    session = requests.Session()
    session.headers.update(headers)

    probe = session.get(url, headers={"Range": "bytes=0-0"}, stream=True,
                        timeout=60)
    if probe.status_code == 404:
        probe.close()
        raise FileNotFoundError(url)
    if probe.status_code != 206:
        probe.close()
        raise _RangesUnsupported(f"status {probe.status_code}")
    content_range = probe.headers.get("Content-Range", "")
    probe.close()
    try:
        total = int(content_range.split("/")[-1])
    except (ValueError, IndexError):
        raise _RangesUnsupported("no Content-Range total")

    spans = [(off, min(chunk_bytes, total - off))
             for off in range(0, total, chunk_bytes)]

    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as fh:
        fh.truncate(total)

    bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
               desc=desc or dest.name, leave=False)
    fd = os.open(tmp, os.O_WRONLY)
    try:
        def fetch_span(span):
            offset, length = span
            last = None
            for attempt in range(retries):
                written = 0
                try:
                    span_headers = {"Range":
                                    f"bytes={offset}-{offset + length - 1}"}
                    with session.get(url, headers=span_headers, stream=True,
                                     timeout=120) as resp:
                        resp.raise_for_status()
                        for block in resp.iter_content(chunk_size=1 << 20):
                            if not block:
                                continue
                            os.pwrite(fd, block, offset + written)
                            written += len(block)
                            bar.update(len(block))
                        if written != length:
                            raise IOError(
                                f"short range: got {written} of {length}")
                    return
                except Exception as exc:               # noqa: BLE001
                    last = exc
                    # Un-count what this failed attempt added, or the bar
                    # would overshoot once the retry re-sends the same bytes.
                    if written:
                        bar.update(-written)
                    log.debug("chunk %d retry %d: %s", offset, attempt + 1, exc)
            raise IOError(f"chunk at {offset} failed after {retries}: {last}")

        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures = [pool.submit(fetch_span, span) for span in spans]
            for future in as_completed(futures):
                future.result()
    finally:
        os.close(fd)
        bar.close()

    os.replace(tmp, dest)
    return dest


def _unlink(path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
