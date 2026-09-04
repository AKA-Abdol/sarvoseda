"""``vcprep`` - command line for the preprocessing pipeline.

    vcprep run          fetch -> prefilter -> UVR -> silence -> quality -> split
    vcprep plan         show the work queue without doing anything
    vcprep stage        run one stage over existing manifest records
    vcprep calibrate    read score distributions, suggest thresholds
    vcprep rescore      re-apply thresholds to existing scores (no audio read)
    vcprep stats        where the corpus went, and why
    vcprep fetch-models download DNSMOS weights
    vcprep init-config  write a commented YAML you can edit
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from .config import DatasetSource, PipelineConfig
from .manifest import Manifest, ManifestStore
from .runner import STAGE_ORDER, Pipeline

log = logging.getLogger("vcprep")


# ---------------------------------------------------------------------------
# argument wiring
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vcprep",
        description="Speech dataset preprocessing: HF download -> UVR vocal "
                    "isolation -> silence removal -> quality scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument("--config", help="pipeline YAML (CLI flags win over it)")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # Repeated on every subcommand (with SUPPRESS so they only apply when
    # actually typed) so both `vcprep --config X run` and `vcprep run
    # --config X` work - the second is what people reach for.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS,
                        help="pipeline YAML (CLI flags win over it)")
    common.add_argument("--log-level", default=argparse.SUPPRESS,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- shared option groups ----
    def add_common(p):
        p.add_argument("--work-dir", help="scratch space (default: work)")
        p.add_argument("--out-dir", help="final output root (default: out)")
        p.add_argument("--manifest-dir", help="manifest partitions "
                                              "(default: <work-dir>/manifest)")

    def add_source_opts(p):
        p.add_argument("--repo", action="append", default=[], metavar="SPEC",
                       help="HF dataset to ingest; repeatable. "
                            "SPEC = owner/name[@revision][#output-slug]")
        p.add_argument("--sources", metavar="YAML",
                       help="YAML file holding a list of dataset sources")
        p.add_argument("--only", action="append", default=[], metavar="SLUG",
                       help="restrict the run to these configured sources")
        p.add_argument("--shards", help="shard selection, e.g. 1-4,7,9")
        p.add_argument("--limit", type=int, default=None,
                       help="max utterances per source (0 = all)")
        p.add_argument("--limit-per-shard", type=int, default=None)
        p.add_argument("--hf-token", default=None)

    def add_model_opts(p):
        p.add_argument("--uvr-model", metavar="PATH",
                       help="path to a separation model "
                            "(e.g. Kim_Vocal_2.onnx); overrides --model-name")
        p.add_argument("--model-name", metavar="NAME",
                       help="audio-separator model, downloaded if absent "
                            "(default: vocals_mel_band_roformer.ckpt)")
        p.add_argument("--model-dir", metavar="PATH",
                       help="where separation models are kept")
        p.add_argument("--dnsmos-dir", metavar="PATH",
                       help="directory holding sig_bak_ovr.onnx")
        p.add_argument("--nisqa-repo", metavar="PATH")
        p.add_argument("--nisqa-weights", metavar="PATH")
        p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)

    def add_download_opts(p):
        p.add_argument("--download-mode", choices=["stream", "parallel"],
                       default=None,
                       help="'parallel' uses many ranged connections and is "
                            "usually several times faster on a long-haul link "
                            "(default). Run `vcprep netcheck` to measure.")
        p.add_argument("--connections", type=int, default=None,
                       help="simultaneous connections for parallel downloads "
                            "(default 8)")

    def add_exec_opts(p):
        p.add_argument("--backend", choices=["serial", "process", "ray"],
                       default=None,
                       help="execution backend for the CPU stages "
                            "(default: process). 'ray' only pays off across "
                            "several machines.")
        p.add_argument("--num-workers", type=int, default=None,
                       help="backend workers; 0 = one per CPU core")
        p.add_argument("--ray-address", default=None,
                       help="existing Ray cluster to attach to")
        p.add_argument("--no-batch-clips", action="store_true",
                       help="separate one clip per call (slower; for debugging)")
        p.add_argument("--batch-seconds", type=float, default=None,
                       help="audio packed into each separation call "
                            "(default 300)")

    def add_toggle_opts(p):
        p.add_argument("--stages", metavar="LIST",
                       help="comma-separated subset of: " + ",".join(STAGE_ORDER))
        p.add_argument("--no-prefilter", action="store_true")
        p.add_argument("--no-separate", action="store_true")
        p.add_argument("--no-vad", action="store_true")
        p.add_argument("--no-quality", action="store_true")
        p.add_argument("--no-materialize", action="store_true")
        p.add_argument("--with-nisqa", action="store_true",
                       help="enable NISQA (adds the discontinuity axis)")
        p.add_argument("--with-squim", action="store_true",
                       help="enable TorchAudio SQUIM (predicted PESQ/STOI)")
        p.add_argument("--keep-intermediates", action="store_true",
                       help="do not delete per-stage scratch audio")

    # ---- run ----
    run = sub.add_parser("run", parents=[common], help="run the full pipeline over the queue")
    add_common(run); add_source_opts(run); add_model_opts(run)
    add_download_opts(run); add_exec_opts(run); add_toggle_opts(run)

    # ---- plan ----
    plan = sub.add_parser("plan", parents=[common], help="print the work queue and exit")
    add_common(plan); add_source_opts(plan)

    # ---- stage ----
    stage = sub.add_parser("stage", parents=[common], help="run a single stage over the manifest")
    stage.add_argument("name", choices=STAGE_ORDER)
    add_common(stage); add_source_opts(stage); add_model_opts(stage)
    add_download_opts(stage); add_exec_opts(stage)
    stage.add_argument("--with-nisqa", action="store_true")
    stage.add_argument("--with-squim", action="store_true")

    # ---- rescore ----
    rs = sub.add_parser("rescore", parents=[common],
                        help="re-apply thresholds to existing scores "
                             "(no audio is read)")
    add_common(rs)
    rs.add_argument("--source", help="restrict to one source slug")
    rs.add_argument("--dry-run", action="store_true",
                    help="report what would change and stop")

    # ---- calibrate ----
    net = sub.add_parser("netcheck", parents=[common],
                         help="diagnose slow downloads: your link, Hugging "
                              "Face, or single-stream throughput?")
    add_source_opts(net)
    net.add_argument("--connections", type=int, default=8,
                     help="connections for the parallel probe (default 8)")
    net.add_argument("--probe-mb", type=int, default=32,
                     help="megabytes per probe (default 32)")
    net.add_argument("--skip-baseline", action="store_true",
                     help="skip the neutral-CDN probe")

    cal = sub.add_parser("calibrate", parents=[common], help="score distributions and thresholds")
    add_common(cal)
    cal.add_argument("--source", help="restrict to one source slug")
    cal.add_argument("--target-keep", type=float, default=0.80,
                     help="fraction to retain per axis (default 0.80)")

    # ---- stats ----
    st = sub.add_parser("stats", parents=[common], help="summarise the manifest")
    add_common(st)
    st.add_argument("--json", action="store_true")

    # ---- fetch-models ----
    fm = sub.add_parser("fetch-models", parents=[common], help="download DNSMOS weights")
    fm.add_argument("--dnsmos-dir", default="models/dnsmos")

    # ---- init-config ----
    ic = sub.add_parser("init-config", parents=[common], help="write a starter YAML")
    ic.add_argument("path", nargs="?", default="configs/pipeline.yaml")

    return parser


_EPILOG = """
examples:
  # smoke test: 20 clips from one shard, everything on, on this laptop
  vcprep run --limit-per-shard 20 --shards 1 \\
             --uvr-model "/Applications/Ultimate Vocal Remover.app/Contents/\\
Resources/models/MDX_Net_Models/Kim_Vocal_2.onnx"

  # queue several HF datasets; outputs are foldered and named per repo
  vcprep run --repo MohammadGholizadeh/filimo-farsi-raw \\
             --repo mozilla-foundation/common_voice_17_0#cv17 \\
             --out-dir /data/out --work-dir /scratch

  # re-score and re-split without re-separating anything
  vcprep stage quality --with-nisqa && vcprep stage materialize
"""


# ---------------------------------------------------------------------------
def parse_shards(spec: Optional[str]) -> Optional[List[int]]:
    """``"1-4,7,9"`` -> ``[1,2,3,4,7,9]``."""
    if not spec:
        return None
    out: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def build_config(args) -> PipelineConfig:
    cfg = PipelineConfig.load(getattr(args, "config", None))

    # ---- sources: --sources YAML, then --repo, replace the default queue ----
    sources: List[DatasetSource] = []
    if getattr(args, "sources", None):
        import yaml
        with open(args.sources, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or []
        entries = data.get("sources", data) if isinstance(data, dict) else data
        for entry in entries:
            if isinstance(entry, str):
                sources.append(DatasetSource.from_spec(entry))
            else:
                src = DatasetSource()
                for key, value in entry.items():
                    if not hasattr(src, key):
                        raise KeyError(f"unknown source key: {key!r}")
                    setattr(src, key, value)
                sources.append(src)
    for spec in getattr(args, "repo", []) or []:
        sources.append(DatasetSource.from_spec(spec))
    if sources:
        cfg.sources = sources

    shards = parse_shards(getattr(args, "shards", None))
    for src in cfg.sources:
        if shards is not None:
            src.shards = shards
        if getattr(args, "limit", None) is not None:
            src.limit = args.limit
        if getattr(args, "limit_per_shard", None) is not None:
            src.limit_per_shard = args.limit_per_shard
        if getattr(args, "hf_token", None):
            src.hf_token = args.hf_token

    # ---- paths ----
    if getattr(args, "work_dir", None):
        cfg.paths.work_dir = args.work_dir
        cfg.paths.manifest_dir = ""
    if getattr(args, "out_dir", None):
        cfg.paths.out_dir = args.out_dir
    if getattr(args, "manifest_dir", None):
        cfg.paths.manifest_dir = args.manifest_dir
    cfg.paths.resolve()

    # ---- models ----
    if getattr(args, "model_name", None):
        cfg.separate.model_name = args.model_name
    if getattr(args, "model_dir", None):
        cfg.separate.model_dir = args.model_dir
    if getattr(args, "uvr_model", None):
        cfg.separate.model_path = args.uvr_model
    cfg.separate.resolve()
    if getattr(args, "dnsmos_dir", None):
        cfg.quality.dnsmos_dir = args.dnsmos_dir
    if getattr(args, "nisqa_repo", None):
        cfg.quality.nisqa_repo = args.nisqa_repo
    if getattr(args, "nisqa_weights", None):
        cfg.quality.nisqa_weights = args.nisqa_weights
    if getattr(args, "device", None):
        cfg.quality.device = args.device

    # ---- stage toggles ----
    for flag, section in (("no_prefilter", "prefilter"),
                          ("no_separate", "separate"),
                          ("no_vad", "vad"),
                          ("no_quality", "quality"),
                          ("no_materialize", "materialize")):
        if getattr(args, flag, False):
            getattr(cfg, section).enabled = False
    if getattr(args, "with_nisqa", False):
        cfg.quality.use_nisqa = True
    if getattr(args, "with_squim", False):
        cfg.quality.use_squim = True

    # ---- downloads ----
    if getattr(args, "download_mode", None):
        cfg.fetch.download_mode = args.download_mode
    if getattr(args, "connections", None) is not None:
        cfg.fetch.connections = args.connections

    # ---- execution ----
    if getattr(args, "backend", None):
        cfg.backend = args.backend
    if getattr(args, "num_workers", None) is not None:
        cfg.num_workers = args.num_workers
    if getattr(args, "no_batch_clips", False):
        cfg.separate.batch_clips = False
    if getattr(args, "batch_seconds", None) is not None:
        cfg.separate.batch_seconds = args.batch_seconds

    if getattr(args, "log_level", None):
        cfg.log_level = args.log_level
    return cfg


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_run(args) -> int:
    cfg = build_config(args)
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None
    pipeline = Pipeline(cfg, stages=stages, backend=cfg.backend,
                        num_workers=cfg.num_workers,
                        ray_address=getattr(args, "ray_address", None))
    results = pipeline.run(
        only_sources=args.only or None,
        only_shards=parse_shards(args.shards),
        keep_intermediates=args.keep_intermediates,
    )
    print(json.dumps(pipeline.store.summary(), indent=2, ensure_ascii=False))
    return 0 if results else 1


def cmd_plan(args) -> int:
    cfg = build_config(args)
    pipeline = Pipeline(cfg)
    units = pipeline.plan(args.only or None)
    shards = parse_shards(args.shards)
    if shards is not None:
        units = [u for u in units if u.shard in set(shards)]
    print(f"{len(units)} work unit(s):")
    for unit in units:
        print(f"  {unit.label:<40} repo={unit.source.repo_id}")
    return 0


def cmd_stage(args) -> int:
    cfg = build_config(args)
    pipeline = Pipeline(cfg, stages=[args.name], backend=cfg.backend,
                        num_workers=cfg.num_workers,
                        ray_address=getattr(args, "ray_address", None))
    node = pipeline.nodes[args.name]
    shards = parse_shards(args.shards)
    only = args.only or [s.slug for s in cfg.active_sources()]

    totals = {}
    if args.name == "fetch":
        node.store = pipeline.store
        for unit in pipeline.plan(args.only or None):
            if shards is not None and unit.shard not in set(shards):
                continue
            node.bind(pipeline.store.partition(unit.source.slug, unit.shard))
            added, skipped = node.fetch_unit(unit)
            totals[unit.label] = {"added": added, "skipped": skipped}
    else:
        from .runner import PARALLEL_STAGES
        for slug in only:
            # No shard list given: replay every partition this source has.
            unit_shards = shards if shards is not None else _shards_of(
                pipeline.store, slug)
            for shard in unit_shards:
                node.bind(pipeline.store.partition(slug, shard),
                          pipeline.backend if args.name in PARALLEL_STAGES
                          else None)
                if args.name == "materialize":
                    node.store = pipeline.store
                key = f"{slug}/{shard if shard is not None else 'all'}"
                totals[key] = node.run(shard=shard, source=slug)
    pipeline.store.flush_all()
    pipeline.backend.close()
    print(json.dumps(totals, indent=2, ensure_ascii=False))
    return 0


def _shards_of(store, slug: str) -> List[Optional[int]]:
    """Shard ids that already have a manifest partition for this source."""
    found = set()
    for path in store.partitions():
        if path.parent.name != slug:
            continue
        stem = path.stem
        found.add(int(stem.split("_")[-1]) if stem.startswith("shard_") else None)
    return sorted(found, key=lambda v: (v is None, v)) or [None]


def cmd_rescore(args) -> int:
    """Re-decide keep vs low_quality from metrics already in the manifest.

    Thresholds are applied during the quality stage, so changing them in the
    config would otherwise require re-scoring the corpus. This re-runs only the
    decision, reading no audio, which makes threshold tuning a seconds-long
    loop: calibrate -> edit config -> rescore -> materialize.
    """
    from .manifest import KEEP, LOW_QUALITY, Manifest
    from .stages import decide_quality

    cfg = build_config(args)
    store = ManifestStore(cfg.paths.manifest_dir)

    moved = {"keep->low_quality": 0, "low_quality->keep": 0, "unchanged": 0}
    scored = 0
    for path in store.partitions():
        manifest = Manifest(str(path))
        dirty = False
        for rec in manifest.records():
            if args.source and rec.source != args.source:
                continue
            # Only records the quality stage actually judged. Anything dropped
            # earlier (no speech, too short) stays dropped - those verdicts are
            # not threshold-dependent.
            if rec.status not in (KEEP, LOW_QUALITY):
                continue
            if "dnsmos_ovrl" not in rec.metrics and "bandwidth_hz" not in rec.metrics:
                continue
            scored += 1
            before = rec.status
            # Drop the previous run's quality labels before re-deciding.
            for label in rec.metrics.get("quality_failures", []):
                if label in rec.reasons:
                    rec.reasons.remove(label)
            decide_quality(rec, cfg)
            if rec.status == before:
                moved["unchanged"] += 1
            else:
                moved[f"{before}->{rec.status}"] += 1
                dirty = True
            manifest.update(rec)
        if dirty and not args.dry_run:
            manifest.flush()

    print(f"scored records examined : {scored}")
    for key, count in moved.items():
        print(f"  {key:<22} {count}")
    if args.dry_run:
        print("\n(dry run - nothing written)")
    elif moved["keep->low_quality"] or moved["low_quality->keep"]:
        print("\nNow run `vcprep stage materialize` to move the files.")
    return 0


def cmd_netcheck(args) -> int:
    from . import netcheck

    cfg = build_config(args)
    sources = cfg.active_sources()
    if not sources:
        print("no sources configured")
        return 1
    src = sources[0]

    url = _probe_url(src)
    if url is None:
        print(f"cannot build a probe URL for layout {src.layout!r}")
        return 1

    token = src.hf_token or os.environ.get("HF_TOKEN")
    print(f"probing {src.repo_id}")
    results = netcheck.run(url, token=token,
                           connections=args.connections,
                           size=args.probe_mb * 1024 * 1024,
                           skip_baseline=args.skip_baseline)
    netcheck.report(results)
    return 0


def _probe_url(src) -> Optional[str]:
    """Pick a real, large file in the repo to measure against."""
    from huggingface_hub import HfApi, hf_hub_url

    try:
        api = HfApi(token=src.hf_token or os.environ.get("HF_TOKEN"))
        info = api.repo_info(src.repo_id, repo_type="dataset",
                             revision=src.revision, files_metadata=True)
        biggest = max((f for f in info.siblings if (f.size or 0) > 0),
                      key=lambda f: f.size or 0, default=None)
        if biggest is not None:
            return hf_hub_url(repo_id=src.repo_id, filename=biggest.rfilename,
                              repo_type="dataset", revision=src.revision)
    except Exception as exc:
        log.debug("could not list repo files: %s", exc)
    return None


def cmd_calibrate(args) -> int:
    from . import calibrate as cal

    cfg = build_config(args)
    manifest = ManifestStore(cfg.paths.manifest_dir).as_manifest()
    pools = cal.collect(manifest, source=args.source)
    if not pools:
        print("no scored records yet - run `vcprep stage quality` first.")
        return 1
    print(cal.report(pools, target_keep=args.target_keep))

    thresholds = {
        "dnsmos_ovrl": cfg.quality.min_ovrl,
        "dnsmos_sig": cfg.quality.min_sig,
        "dnsmos_bak": cfg.quality.min_bak,
        "bandwidth_hz": cfg.quality.min_bandwidth_hz,
    }
    rate = cal.joint_keep_rate(manifest, thresholds, source=args.source)
    print(f"\ncurrent configured thresholds would keep {rate:.1%} of scored clips")
    return 0


def cmd_stats(args) -> int:
    cfg = build_config(args)
    summary = ManifestStore(cfg.paths.manifest_dir).summary()
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    print(f"total records : {summary['total']}")
    print(f"clean hours   : {summary['keep_hours']}")
    print("\nby status:")
    for key, value in sorted(summary["by_status"].items()):
        print(f"  {key:<14} {value}")
    print("\nby source:")
    for slug, bucket in sorted(summary["by_source"].items()):
        print(f"  {slug:<28} {bucket['keep']}/{bucket['total']} kept "
              f"({bucket['keep_hours']} h)")
    if summary["reject_reasons"]:
        print("\nwhy clips were dropped or demoted:")
        for reason, count in summary["reject_reasons"].items():
            print(f"  {reason:<28} {count}")
    return 0


def cmd_fetch_models(args) -> int:
    from .metrics.dnsmos import download_model
    path = download_model(args.dnsmos_dir)
    print(f"DNSMOS ready: {path}")
    print("NISQA (optional, adds the discontinuity axis): bash scripts/setup_nisqa.sh")
    return 0


def cmd_init_config(args) -> int:
    dest = Path(args.path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    PipelineConfig().dump(str(dest))
    print(f"wrote {dest}")
    return 0


COMMANDS = {
    "run": cmd_run,
    "plan": cmd_plan,
    "stage": cmd_stage,
    "rescore": cmd_rescore,
    "calibrate": cmd_calibrate,
    "stats": cmd_stats,
    "netcheck": cmd_netcheck,
    "fetch-models": cmd_fetch_models,
    "init-config": cmd_init_config,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(args, "log_level", None) or "INFO",
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
