"""``vcft`` - command line for seed-vc v2 fine-tuning.

Separate from ``vcprep`` on purpose: preprocessing and training have different
dependency stacks, different hardware profiles and different failure modes, and
one should be re-runnable without dragging in the other.

    vcft prepare    pipeline output -> seed-vc dataset directory
    vcft train      launch train_v2.py through accelerate
    vcft status     what has been prepared and what has been trained
    vcft init-config
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import FinetuneConfig

log = logging.getLogger("vcft")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vcft",
        description="Fine-tune seed-vc v2 on the preprocessed corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument("--config", help="finetune YAML")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # Repeated on every subcommand so `vcft train --config X` works as well
    # as `vcft --config X train`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS,
                        help="finetune YAML")
    common.add_argument("--log-level", default=argparse.SUPPRESS,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- prepare ----
    prep = sub.add_parser("prepare", parents=[common], help="build the seed-vc dataset directory")
    prep.add_argument("--clean-dir", help="pipeline clean output (out/clean)")
    prep.add_argument("--dataset-dir", help="where to assemble the training set")
    prep.add_argument("--source", action="append", default=[], metavar="SLUG",
                      help="restrict to these dataset slugs; repeatable")
    prep.add_argument("--include-low-quality", action="store_true")
    prep.add_argument("--max-files", type=int, default=None)
    prep.add_argument("--min-duration", type=float, default=None)
    prep.add_argument("--max-duration", type=float, default=None)
    prep.add_argument("--min-dnsmos-ovrl", type=float, default=None,
                      help="extra quality floor on top of the pipeline's own")
    prep.add_argument("--min-dnsmos-bak", type=float, default=None)
    prep.add_argument("--val-fraction", type=float, default=None)
    prep.add_argument("--materialize", choices=["link", "symlink", "copy"],
                      default=None)

    # ---- train ----
    train = sub.add_parser("train", parents=[common], help="launch seed-vc v2 fine-tuning")
    train.add_argument("--dataset-dir", help="override the prepared dataset")
    train.add_argument("--repo-dir", help="path to the seed-vc checkout")
    train.add_argument("--run-name")
    train.add_argument("--seedvc-config", dest="seedvc_config",
                       help="config path *inside* the seed-vc repo")
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--max-epochs", type=int, default=None)
    train.add_argument("--save-every", type=int, default=None)
    train.add_argument("--num-workers", type=int, default=None)
    train.add_argument("--num-processes", type=int, default=None,
                       help="GPUs for accelerate")
    train.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"],
                       default=None)
    train.add_argument("--train-ar", action="store_true",
                       help="also fine-tune the AR stage (needs much more data)")
    train.add_argument("--no-train-cfm", action="store_true")
    train.add_argument("--pretrained-cfm-ckpt", default=None)
    train.add_argument("--pretrained-ar-ckpt", default=None)
    train.add_argument("--dry-run", action="store_true",
                       help="print the command instead of running it")

    # ---- status ----
    status = sub.add_parser("status", parents=[common], help="prepared data and checkpoints")
    status.add_argument("--dataset-dir")
    status.add_argument("--repo-dir")
    status.add_argument("--run-name")

    # ---- init-config ----
    init = sub.add_parser("init-config", parents=[common], help="write a starter YAML")
    init.add_argument("path", nargs="?", default="configs/finetune.yaml")
    return parser


_EPILOG = """
examples:
  vcft prepare --clean-dir out/clean --dataset-dir seedvc_data
  vcft prepare --source filimo-farsi-raw --min-dnsmos-ovrl 3.1 --max-files 20000
  vcft train --num-processes 2 --batch-size 8 --max-steps 40000
  vcft train --dry-run          # show the accelerate command and stop
"""


def build_config(args) -> FinetuneConfig:
    cfg = FinetuneConfig.load(getattr(args, "config", None))
    data, train = cfg.data, cfg.train

    for attr, target in (("clean_dir", "clean_dir"), ("dataset_dir", "dataset_dir"),
                         ("max_files", "max_files"),
                         ("min_duration", "min_duration"),
                         ("max_duration", "max_duration"),
                         ("min_dnsmos_ovrl", "min_dnsmos_ovrl"),
                         ("min_dnsmos_bak", "min_dnsmos_bak"),
                         ("val_fraction", "val_fraction"),
                         ("materialize", "materialize")):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(data, target, value)
    if getattr(args, "source", None):
        data.sources = args.source
    if getattr(args, "include_low_quality", False):
        data.include_low_quality = True

    for attr, target in (("repo_dir", "repo_dir"), ("run_name", "run_name"),
                         ("seedvc_config", "config"),
                         ("batch_size", "batch_size"), ("max_steps", "max_steps"),
                         ("max_epochs", "max_epochs"),
                         ("save_every", "save_every"),
                         ("num_workers", "num_workers"),
                         ("num_processes", "num_processes"),
                         ("mixed_precision", "mixed_precision"),
                         ("pretrained_cfm_ckpt", "pretrained_cfm_ckpt"),
                         ("pretrained_ar_ckpt", "pretrained_ar_ckpt")):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(train, target, value)
    if getattr(args, "train_ar", False):
        train.train_ar = True
    if getattr(args, "no_train_cfm", False):
        train.train_cfm = False

    if getattr(args, "log_level", None):
        cfg.log_level = args.log_level
    return cfg


def cmd_prepare(args) -> int:
    from . import prepare

    cfg = build_config(args)
    summary = prepare.build(cfg.data)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_train(args) -> int:
    from . import train as trainer

    cfg = build_config(args)
    return trainer.run(cfg, dataset_dir=args.dataset_dir, dry_run=args.dry_run)


def cmd_status(args) -> int:
    from . import train as trainer

    cfg = build_config(args)
    dataset_dir = Path(args.dataset_dir or cfg.data.dataset_dir)
    summary_path = dataset_dir / "dataset_summary.json"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
    else:
        print(f"no prepared dataset at {dataset_dir} (run `vcft prepare`)")

    found = trainer.checkpoints(cfg)
    if found:
        print(f"\ncheckpoints in runs/{cfg.train.run_name}:")
        for path in found:
            size = path.stat().st_size / (1024 ** 2)
            print(f"  {path.name:<44} {size:8.1f} MB")
    else:
        print(f"\nno checkpoints yet for run {cfg.train.run_name!r}")
    return 0


def cmd_init_config(args) -> int:
    FinetuneConfig().dump(args.path)
    print(f"wrote {args.path}")
    return 0


COMMANDS = {
    "prepare": cmd_prepare,
    "train": cmd_train,
    "status": cmd_status,
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
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
