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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vcprep.audio import load_audio, probe                # noqa: E402
from vcprep.config import PipelineConfig                  # noqa: E402
from vcprep.manifest import Manifest, Record              # noqa: E402
from vcprep.metrics.dnsmos import DNSMOS                   # noqa: E402
from vcprep.nodes.separate import SeparateNode             # noqa: E402

AXES = ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl")


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
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    files = sorted(p for p in Path(args.audio_dir).iterdir()
                   if p.suffix.lower() in {".mp3", ".wav", ".flac"})[: args.limit]
    if not files:
        print(f"no audio in {args.audio_dir}")
        return 1
    total_audio = sum(probe(str(f))[0] for f in files)

    base = PipelineConfig.load(args.config)
    mos = DNSMOS(base.quality.dnsmos_dir, device="cpu")

    tmp = Path(tempfile.mkdtemp(prefix="vcmp_"))
    results = {}
    try:
        for model in args.model:
            label = Path(model).name
            print(f"running {label} over {len(files)} clips "
                  f"({total_audio:.0f}s of audio) ...")
            cfg = PipelineConfig.load(args.config)
            outputs, elapsed = run_model(cfg, model, files, tmp / label)
            scores = {axis: [] for axis in AXES}
            for path in outputs.values():
                audio, sr = load_audio(path, mono=True)
                got = mos.score(audio, sr)
                for axis in AXES:
                    if got[axis] == got[axis]:
                        scores[axis].append(got[axis])
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
