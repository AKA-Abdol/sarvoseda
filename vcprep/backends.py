"""Execution backends for the CPU-bound stages.

Only prefilter, VAD, quality and materialize go through here. Separation
deliberately does not: it owns the GPU, and running several copies of it would
multiply VRAM rather than throughput. Its speedup comes from batching instead
(see :mod:`vcprep.nodes.separate`).

Three implementations behind one interface:

``serial``   one process, no pickling. The baseline, and the easiest to debug.
``process``  ProcessPoolExecutor over local cores. The right default on a
             single server: DNSMOS and Silero are per-process singletons, so a
             worker loads them once and then amortises over thousands of clips.
``ray``      the same map, distributed. Worth it only across machines; on one
             box it adds a head node and an object store for no gain.

Workers are addressed by *name* rather than by closure so the process backend
can pickle the call, and the config is handed to each worker once at
initialisation instead of riding along with every task.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from tqdm import tqdm

from .config import PipelineConfig
from .manifest import Record

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# worker registry - names must resolve identically in a fresh process
# --------------------------------------------------------------------------
_WORKER_CFG: Optional[PipelineConfig] = None
_WORKER_EXTRA: Dict[str, Any] = {}


def init_worker(cfg_dict: Dict[str, Any], extra: Dict[str, Any],
                limit_threads: bool = True) -> None:
    """Runs once per worker process (or once in-process for the serial path)."""
    global _WORKER_CFG, _WORKER_EXTRA
    from .config import PipelineConfig, _merge

    _WORKER_CFG = _merge(PipelineConfig(), cfg_dict)
    _WORKER_EXTRA = extra or {}
    if not limit_threads:
        # The serial backend is the whole process: leave it every core.
        return
    # Pool workers must not each render progress bars or duplicate logs.
    logging.basicConfig(level=logging.ERROR)
    # BLAS and torch otherwise each grab every core, so N workers oversubscribe
    # the machine badly and run slower than one.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def run_worker(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one task in a worker. ``payload`` is a serialised Record."""
    from . import stages

    cfg = _WORKER_CFG
    rec = Record(**payload)
    if name == "prefilter":
        rec = stages.prefilter_record(rec, cfg)
    elif name == "vad":
        rec = stages.vad_record(rec, cfg, _WORKER_EXTRA["out_dir"])
    elif name == "quality":
        rec = stages.quality_record(rec, cfg)
    else:
        raise KeyError(f"unknown worker: {name!r}")
    return rec.__dict__


# --------------------------------------------------------------------------
class Backend(ABC):
    name = "base"

    def __init__(self, cfg: PipelineConfig, num_workers: int = 0):
        self.cfg = cfg
        self.num_workers = num_workers or (os.cpu_count() or 1)

    @abstractmethod
    def map(self, worker: str, records: List[Record],
            extra: Optional[Dict[str, Any]] = None,
            desc: str = "") -> List[Record]:
        """Apply ``worker`` to every record, returning them updated."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SerialBackend(Backend):
    name = "serial"

    def __init__(self, cfg: PipelineConfig, num_workers: int = 0):
        super().__init__(cfg, 1)
        self.num_workers = 1

    def map(self, worker: str, records, extra=None, desc="") -> List[Record]:
        init_worker(self.cfg.to_dict(), extra or {}, limit_threads=False)
        out = []
        for rec in tqdm(records, desc=desc or worker, unit="clip", leave=False):
            out.append(Record(**run_worker(worker, rec.__dict__)))
        return out


#: Below this many records, spawning a pool costs more than it saves. Each
#: worker is a fresh interpreter that imports torch and loads Silero and
#: DNSMOS; measured on 12-clip units, a 4-worker pool ran ~55% *slower* than
#: serial. Real shards hold thousands of clips, where the pool wins decisively.
MIN_PARALLEL_ITEMS = 256


class ProcessBackend(Backend):
    """Local process pool. The default on a single server."""
    name = "process"

    def map(self, worker: str, records, extra=None, desc="") -> List[Record]:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        if not records:
            return []
        workers = max(1, min(self.num_workers, len(records)))
        if workers == 1 or len(records) < MIN_PARALLEL_ITEMS:
            log.debug("%d records: running %s serially (pool startup would "
                      "cost more than it saves)", len(records), worker)
            return SerialBackend(self.cfg).map(worker, records, extra, desc)

        payloads = [r.__dict__ for r in records]
        # 'spawn' rather than 'fork': torch and onnxruntime both hold state
        # that does not survive a fork safely, and a forked CUDA context is a
        # hard error.
        ctx = mp.get_context("spawn")

        cfg_dict = self.cfg.to_dict()
        # Every worker builds its own DNSMOS session, so on 'auto' with CUDA
        # present each one opens its own CUDA context - hundreds of MB apiece,
        # all contending - for a model that gains little from the GPU (a
        # typical clip is one 9 s window at batch size 1: latency-bound, not
        # compute-bound). With plenty of cores the CPU pool wins outright.
        # With only a few, it cannot keep up and the GPU is the better home,
        # so pin to CPU only past max_gpu_workers. Explicit 'cuda' or 'cpu'
        # is always obeyed.
        quality = cfg_dict.get("quality", {})
        limit = quality.get("max_gpu_workers", 4)
        if quality.get("device") == "auto" and workers > limit:
            quality["device"] = "cpu"
            log.info("%d workers (> max_gpu_workers=%d): running DNSMOS on CPU. "
                     "Force the GPU with --device cuda.", workers, limit)
        chunk = max(1, len(payloads) // (workers * 8))

        out: List[Record] = []
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx,
            initializer=init_worker,
            initargs=(cfg_dict, extra or {}),
        ) as pool:
            futures = pool.map(_apply, ((worker, p) for p in payloads),
                               chunksize=chunk)
            for result in tqdm(futures, total=len(payloads),
                               desc=f"{desc or worker} x{workers}",
                               unit="clip", leave=False):
                out.append(Record(**result))
        return out


def _apply(args) -> Dict[str, Any]:
    """Top-level so ProcessPoolExecutor can pickle it."""
    name, payload = args
    return run_worker(name, payload)


class RayBackend(Backend):
    """Distributed map. Only worth its overhead across several machines."""
    name = "ray"

    def __init__(self, cfg: PipelineConfig, num_workers: int = 0,
                 address: Optional[str] = None):
        super().__init__(cfg, num_workers)
        self.address = address
        self._ray = None

    def _ensure(self):
        if self._ray is not None:
            return self._ray
        try:
            import ray
        except ImportError as exc:
            raise RuntimeError(
                "the ray backend needs Ray installed:  pip install 'ray[default]'\n"
                "On a single machine, --backend process is equivalent and lighter."
            ) from exc
        if not ray.is_initialized():
            ray.init(address=self.address, ignore_reinit_error=True,
                     log_to_driver=False)
        self._ray = ray
        return ray

    def map(self, worker: str, records, extra=None, desc="") -> List[Record]:
        if not records:
            return []
        ray = self._ensure()

        cfg_dict = self.cfg.to_dict()
        extra = extra or {}

        # An actor pool, not stateless tasks: actors keep DNSMOS and Silero
        # resident between calls, which is the whole point on a large corpus.
        @ray.remote
        class _Worker:
            def __init__(self, cfg_dict, extra):
                init_worker(cfg_dict, extra)

            def run(self, name, payloads):
                return [run_worker(name, p) for p in payloads]

        n = max(1, self.num_workers)
        actors = [_Worker.remote(cfg_dict, extra) for _ in range(n)]

        payloads = [r.__dict__ for r in records]
        batch = max(1, len(payloads) // (n * 8))
        chunks = [payloads[i:i + batch] for i in range(0, len(payloads), batch)]

        pending = [actors[i % n].run.remote(worker, chunk)
                   for i, chunk in enumerate(chunks)]

        out: List[Record] = []
        with tqdm(total=len(payloads), desc=f"{desc or worker} ray x{n}",
                  unit="clip", leave=False) as bar:
            while pending:
                done, pending = ray.wait(pending, num_returns=1)
                for result in ray.get(done[0]):
                    out.append(Record(**result))
                    bar.update(1)
        for actor in actors:
            ray.kill(actor, no_restart=True)
        return out

    def close(self) -> None:
        if self._ray is not None and self._ray.is_initialized():
            self._ray.shutdown()
            self._ray = None


BACKENDS = {
    "serial": SerialBackend,
    "process": ProcessBackend,
    "ray": RayBackend,
}


def make_backend(name: str, cfg: PipelineConfig, num_workers: int = 0,
                 address: Optional[str] = None) -> Backend:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; pick one of "
                         f"{sorted(BACKENDS)}")
    if name == "ray":
        return RayBackend(cfg, num_workers, address=address)
    return BACKENDS[name](cfg, num_workers)
