#!/usr/bin/env python
"""Compare separation models on *your* audio, by DNSMOS rather than by SDR.

Published SDR figures are measured on studio music benchmarks (MUSDB and
friends). That is a different regime from lossy, band-limited movie audio, and
a model that wins by 2.4 dB SDR there may win by nothing here - where the
limiting factor is the source material, not the separator's residual.

So measure it. This runs each model over the same clips and reports mean
DNSMOS SIG / BAK / OVRL plus inter-word SNR, with wall-clock cost:

    python scripts/compare_separators.py \\
        --audio-dir work/raw/<slug>/shard_001 \\
        --model vocals_mel_band_roformer.ckpt \\
        --model /path/to/Kim_Vocal_2.onnx \\
        --limit 200

BAK is the axis that matters most here: it measures residual background, which
is exactly what a separator is for.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vcprep.audio import load_audio, probe                # noqa: E402
from vcprep.config import PipelineConfig                  # noqa: E402
from vcprep.manifest import Manifest, Record              # noqa: E402
from vcprep.metrics.dnsmos import DNSMOS                   # noqa: E402
from vcprep.nodes.separate import SeparateNode             # noqa: E402

AXES = ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl")


def engine_of(model: str) -> str:
    """Which runtime a model uses. MDX-Net .onnx -> onnxruntime, else torch."""
    return "onnxruntime" if model.lower().endswith(".onnx") else "torch"


def device_of(engine: str) -> str:
    """The device that engine will really use, right now."""
    if engine == "onnxruntime":
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return "cuda"
        except Exception:
            return "unavailable"
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        return "mps" if (mps and mps.is_available()) else "cpu"
    except Exception:
        return "unavailable"


def run_model(cfg: PipelineConfig, model: str, files, work: Path):
    cfg = PipelineConfig.load(None) if cfg is None else cfg
    cfg.separate.model_path = model if ("/" in model or "\\" in model) else ""
    if not cfg.separate.model_path:
        cfg.separate.model_name = model
    cfg.separate.resolve()
    cfg.fetch.delete_after_separate = False

    manifest = Manifest(str(work / "m.jsonl"))
    for path in files:
        duration, _, _ = probe(str(path))
        manifest.add(Record(uid=Path(path).stem, source="c", shard=1,
                            path=str(path), stage="separate", duration=duration))

    node = SeparateNode(cfg, manifest)
    node.out_dir = work / "vocals"
    node.staging = node.out_dir / "_staging"
    sep = node._load()
    unit = node.unit_dir(node.out_dir, "c", 1)

    started = time.time()
    node._run_batched(sep, manifest.records(), unit)
    elapsed = time.time() - started
    return {r.uid: r.path for r in manifest.records() if r.alive}, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="registry name or path; repeatable")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                    help="device for DNSMOS scoring (default: from config)")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--batch-seconds", type=float, default=None,
                    help="audio packed per separation call; smaller gives more "
                         "frequent progress updates (default 300)")
    ap.add_argument("--force", action="store_true",
                    help="compare even when models run on different devices "
                         "(quality stays valid; timings do not)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    files = sorted(p for p in Path(args.audio_dir).iterdir()
                   if p.suffix.lower() in {".mp3", ".wav", ".flac"})[: args.limit]
    if not files:
        print(f"no audio in {args.audio_dir}")
        return 1
    total_audio = sum(probe(str(f))[0] for f in files)

    base = PipelineConfig.load(args.config)

    # Say up front what each model will run on. An MDX-Net .onnx goes through
    # onnxruntime while a RoFormer .ckpt goes through torch, so one can land on
    # the GPU and the other on the CPU without a word of warning - and then the
    # timing column compares nothing meaningful.
    print(f"{'model':<34} {'engine':<12} {'device':<8}")
    print("-" * 56)
    plan = []
    for model in args.model:
        engine = engine_of(model)
        device = device_of(engine)
        plan.append((model, engine, device))
        print(f"{Path(model).name:<34} {engine:<12} {device:<8}")
    print()

    devices = {d for _, _, d in plan}
    if len(devices) > 1:
        print("!" * 72)
        print("WARNING: these models are not running on the same device, so the")
        print("         speed columns below are NOT comparable.")
        for model, engine, device in plan:
            print(f"           {Path(model).name} -> {engine} on {device}")
        cpu_onnx = [m for m, e, d in plan if e == "onnxruntime" and d == "cpu"]
        if cpu_onnx:
            print()
            print("         onnxruntime has no CUDA provider here, so the .onnx")
            print("         model is on CPU and will be far slower - it is not")
            print("         hung. Fix with the CUDA-matched runtime:")
            print("           python -c \"import torch; print(torch.version.cuda)\"")
            print("           pip uninstall -y onnxruntime onnxruntime-gpu")
            print("           pip install --no-cache-dir 'onnxruntime-gpu<1.27'  # CUDA 12")
            print("         Or run `vcprep doctor`. Quality columns stay valid;")
            print("         only the timings are affected.")
        print("!" * 72)
        print()
        if not args.force:
            print("Re-run with --force to measure anyway (quality only), or fix")
            print("the runtime first for meaningful timings.")
            return 2

    mos = DNSMOS(base.quality.dnsmos_dir, device=args.device or base.quality.device)

    tmp = Path(tempfile.mkdtemp(prefix="vcmp_"))
    results = {}
    try:
        for model, engine, device in plan:
            label = Path(model).name
            print(f"running {label} ({engine} on {device}) over {len(files)} "
                  f"clips, {total_audio:.0f}s of audio ...", flush=True)
            cfg = PipelineConfig.load(args.config)
            if args.batch_seconds is not None:
                cfg.separate.batch_seconds = args.batch_seconds
            outputs, elapsed = run_model(cfg, model, files, tmp / label)
            scores = {axis: [] for axis in AXES}
            skipped = 0
            scored_at = time.time()
            # Scoring used to run silently, which made a stall here look like a
            # hang in separation. Show it.
            for path in tqdm(list(outputs.values()), desc=f"scoring {label}",
                             unit="clip", leave=False):
                try:
                    audio, sr = load_audio(path, mono=True)
                except Exception:
                    skipped += 1
                    continue
                if audio.size == 0:
                    skipped += 1
                    continue
                got = mos.score(audio, sr)
                for axis in AXES:
                    if got[axis] == got[axis]:
                        scores[axis].append(got[axis])
            print(f"  scored {len(scores[AXES[0]])} clips in "
                  f"{time.time() - scored_at:.1f}s"
                  + (f", skipped {skipped} unreadable/empty" if skipped else ""),
                  flush=True)
            results[label] = {
                "n": len(outputs),
                "seconds": elapsed,
                "realtime": total_audio / elapsed if elapsed else float("nan"),
                **{axis: float(np.mean(v)) if v else float("nan")
                   for axis, v in scores.items()},
            }

        print(f"\n{'model':<34} {'n':>4} {'SIG':>7} {'BAK':>7} {'OVRL':>7} "
              f"{'sec':>7} {'xRT':>7}")
        print("-" * 78)
        for label, r in results.items():
            print(f"{label:<34} {r['n']:>4} {r['dnsmos_sig']:7.3f} "
                  f"{r['dnsmos_bak']:7.3f} {r['dnsmos_ovrl']:7.3f} "
                  f"{r['seconds']:7.1f} {r['realtime']:7.2f}")

        if len(results) > 1:
            labels = list(results)
            ref = results[labels[0]]
            print(f"\nrelative to {labels[0]}:")
            for label in labels[1:]:
                r = results[label]
                print(f"  {label:<32} dBAK {r['dnsmos_bak'] - ref['dnsmos_bak']:+.3f}"
                      f"  dOVRL {r['dnsmos_ovrl'] - ref['dnsmos_ovrl']:+.3f}"
                      f"  cost x{ref['realtime'] / r['realtime']:.2f}")
            print("\nA quality delta under ~0.05 is inside run-to-run noise at "
                  "small n. Raise --limit before drawing a conclusion.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
