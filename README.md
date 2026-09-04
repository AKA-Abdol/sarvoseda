# sarvoseda — speech data pipeline + seed-vc v2 fine-tuning

Two decoupled halves:

| | package | CLI | job |
|---|---|---|---|
| 1 | `vcprep` | `vcprep` | HF datasets → UVR → silence removal → quality scoring → `clean/` + `low_quality/` |
| 2 | `seedvc_ft` | `vcft` | that output → seed-vc v2 fine-tuning |

They share nothing but the on-disk layout, so either can be re-run, replaced or
shipped without the other.

```
HF datasets ──► prefilter ──► separation ──────► silence + trim ──► quality ──► clean/
  (queued)      (cheap VAD)   (Mel-Band RoFormer)   (Silero VAD)    (DNSMOS…)   low_quality/
                └─ parallel ─┘  └─ GPU, batched ─┘  └──────── parallel ───────┘
```

---

## Install

> **Two virtualenvs, not one.** seed-vc pins `torch==2.4.0`, `numpy==1.26.4`
> and `transformers==4.46.3`; the separation and VAD models need far newer
> versions of all three. Installed together, whichever went in first breaks.
> This is why the two halves are separate packages.

System packages first — `audio-separator` shells out to ffmpeg for mp3, and
`soundfile` needs libsndfile:

```bash
sudo apt-get install -y ffmpeg libsndfile1 git
```

**Preprocessing:**

```bash
python -m venv .venv-prep && source .venv-prep/bin/activate
bash scripts/setup_server.sh
```

It installs torch **first** from the index matching your driver (later is too
late — a transitive dependency will otherwise pick a build for the wrong CUDA),
then the rest, then `audio-separator[gpu]` + a **CUDA-matched** `onnxruntime-gpu`,
then DNSMOS
weights. Override the CUDA build with `CUDA_TAG=cu118 bash scripts/setup_server.sh`.
It refuses to continue if ffmpeg or libsndfile are missing, and prints the
resolved torch/ONNX providers at the end so you can see the GPU was picked up.

Check what the machine will actually use before starting a long run:

```bash
vcprep doctor
```

It reports the torch device, the onnxruntime providers, and **which engine each
stage runs on** — then diagnoses any mismatch with the exact fix.

<details>
<summary><b>If you see <code>CUDAExecutionProvider not available in ONNXruntime</code></b></summary>

**With a RoFormer model this is usually harmless.** The pipeline uses two
accelerators, and audio-separator's warning only concerns one of them:

| stage | engine | affected? |
|---|---|---|
| separation, RoFormer/MDXC `.ckpt` | **PyTorch** | no |
| separation, MDX-Net `.onnx` | onnxruntime | yes |
| Silero VAD, SQUIM | PyTorch | no |
| DNSMOS | onnxruntime | yes |

audio-separator prints that line at import time regardless of which model you
load. With a RoFormer checkpoint, separation runs on torch — confirm with
`vcprep doctor` or `python -c "import torch; print(torch.cuda.is_available())"`.
The real cost is DNSMOS falling back to CPU. That is survivable — DNSMOS is a
~1 MB model and costs roughly an hour across the whole corpus, less once the
`process` backend spreads it over cores. If matching CUDA versions turns into a
rabbit hole, proceeding on CPU is a perfectly reasonable call.

To fix it, `vcprep doctor` names the cause. The two common ones:

- **CPU-only `onnxruntime` installed on a GPU box** — there is no CUDA provider
  to find. Remove both packages, install the GPU build.
- **`onnxruntime-gpu` installed but the provider fails to load** — typically
  `libcublasLt.so.13: cannot open shared object file`. This is a **CUDA major
  version mismatch** between onnxruntime and torch. onnxruntime loads its CUDA
  libraries lazily and drops to CPU *silently*, so nothing fails at import.

  | onnxruntime-gpu | built for |
  |---|---|
  | ≤ 1.26.0 | CUDA 12 |
  | ≥ 1.27.0 | CUDA 13 |

  Match it to torch. With `torch==2.6.0+cu124` (CUDA 12), a bare
  `pip install onnxruntime-gpu` gets you a CUDA 13 build and breaks:

  ```bash
  python -c "import torch; print(torch.version.cuda)"   # e.g. 12.4
  pip uninstall -y onnxruntime onnxruntime-gpu
  pip install --no-cache-dir "onnxruntime-gpu<1.27"     # CUDA 12
  # (CUDA 13 torch: pip install "onnxruntime-gpu>=1.27")
  ```

  `scripts/setup_server.sh` now detects `torch.version.cuda` and picks the
  matching line automatically.

  To see the loader error onnxruntime swallows:

  ```bash
  python -c "import onnxruntime as o; o.set_default_logger_severity(0); \
      o.InferenceSession('any.onnx', providers=['CUDAExecutionProvider'])"
  ```
</details>

<details>
<summary><b>If you see <code>AttributeError: module 'onnxruntime' has no attribute 'get_available_providers'</code></b></summary>

`onnxruntime` and `onnxruntime-gpu` install into the *same* `onnxruntime/`
directory. Uninstalling one deletes files the other still needs, and pip then
reports the survivor as "already satisfied" and refuses to repair it — leaving a
package that imports but has no attributes. Remove **both**, then reinstall one:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install --force-reinstall --no-cache-dir onnxruntime-gpu
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

If it still fails, delete the directory by hand and reinstall:

```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "$SITE/onnxruntime" "$SITE"/onnxruntime*.dist-info
pip install --no-cache-dir onnxruntime-gpu
```
</details>

**Training** (separate shell, separate venv):

```bash
python -m venv .venv-train && source .venv-train/bin/activate
bash scripts/setup_seedvc.sh
```

It refuses to run if it detects the preprocessing stack in the same
environment. In this venv `vcft` runs as `python -m seedvc_ft.cli` — do *not*
`pip install -e .` here, that would pull `audio-separator` and undo the pins.
`seedvc_ft` imports only stdlib, yaml and soundfile, so it needs nothing else.

The separation model downloads on first use. To use a local UVR model instead:

```bash
vcprep run --uvr-model "…/Ultimate Vocal Remover.app/Contents/Resources/models/MDX_Net_Models/Kim_Vocal_2.onnx"
```

---

## Quick start

```bash
# 1. smoke test — 25 clips from one shard, all stages
vcprep run --shards 1 --limit-per-shard 25

# 2. see the score distribution, pick thresholds from YOUR data
vcprep calibrate --target-keep 0.85

# 3. edit configs/pipeline.yaml, then re-split (seconds — no re-scoring)
vcprep stage materialize

# 4. full run
vcprep run --config configs/pipeline.yaml --out-dir /data/out --work-dir /scratch \
           --backend process --num-workers 16

# 5. train
vcft prepare --clean-dir /data/out/clean --dataset-dir /data/seedvc_data
vcft train --num-processes 2 --batch-size 8 --max-steps 40000
```

---

## Disk

The dataset is ~35 GB of tar shards, but the pipeline never holds more than one
shard. Tars are unpacked **straight from the HTTP socket** — the archive is never
written — and each stage deletes its input once its output is safe. After a
shard finishes, only the manifest and the kept audio remain. Budget a few GB of
scratch regardless of corpus size.

---

## Queuing multiple datasets

Outputs are foldered *and* named by the source repo, so a mixed corpus stays
traceable: `out/clean/<slug>/<slug>__<original>.flac`.

```bash
vcprep run --repo MohammadGholizadeh/filimo-farsi-raw \
           --repo mozilla-foundation/common_voice_17_0#cv17-fa

vcprep run --sources configs/sources.example.yaml
vcprep plan                       # show the queue without doing anything
```

`--repo` spec: `owner/name[@revision][#output-slug]`. Three repo layouts are
auto-detected: **tar** shards, **parquet** (standard HF audio datasets), and
loose **files**.

---

## The manifest is the source of truth

`work/manifest/<source>/shard_NNN.jsonl` holds one record per utterance: every
score, every verdict, every reason. Folders are just a *materialisation* of it.

Manifests are **partitioned per work unit** rather than kept in one global file,
so concurrent units never contend on a single writer and each partition stays
small enough to rewrite atomically on every flush. `stats` and `calibrate` read
across partitions.

That split is the point. Retuning a threshold costs a `vcprep stage materialize`
(seconds), not a re-run of DNSMOS over 400 hours. Every stage is resumable —
interrupt a run and re-issue it; it picks up exactly where it stopped.

```bash
vcprep stats                       # where the corpus went, and why
vcprep calibrate                   # score distributions + suggested cutoffs
vcprep rescore                     # re-apply thresholds, reading no audio
vcprep stage materialize           # move the files to match
vcprep stage quality --with-nisqa  # re-score without re-separating
```

`rescore` is the one that makes threshold tuning cheap. Thresholds are applied
during the quality stage, so changing them in the config would otherwise mean
re-scoring the corpus; `rescore` re-runs only the *decision* over metrics
already in the manifest. Verdicts made earlier (no speech, too short) are not
threshold-dependent and are left alone. The tuning loop is then:

```
calibrate  →  edit configs/pipeline.yaml  →  rescore  →  stage materialize
```

seconds per iteration, on the full corpus.

---

## Slow downloads

The corpus is ~35 GB. Before assuming Hugging Face is at fault, measure — the
fix differs depending on the answer:

```bash
vcprep netcheck
```

It probes a neutral CDN, then HF on one connection, then HF on eight, and tells
you which of three things you have:

- **Neutral CDN is slow too** → the server's link is the problem, not HF.
  Nothing here can fix it; you need a better-connected machine or region.
- **HF is much slower than the CDN** → HF or the route to it. Try a mirror via
  `HF_ENDPOINT`, or run closer to it.
- **Eight connections are much faster than one** → the usual case. A single TCP
  stream is capped by the bandwidth-delay product long before the link
  saturates. Parallel downloading fixes it.

Parallel downloading is the **default**, using ranged requests into a temp file
that is unpacked and deleted per shard. Measured here: **2.92 MB/s single
stream → 9.1 MB/s on 8 connections, 3.11×**, which turns a 4-hour download into
about 1.2 hours.

```bash
vcprep run --download-mode parallel --connections 16   # more on a fat link
vcprep run --download-mode stream                      # zero extra disk
```

Two behaviours worth knowing:

- It costs ~1–1.5 GB of scratch per shard, reclaimed immediately after unpacking.
  `--download-mode stream` unpacks on the fly and uses none, at single-stream speed.
- **Smoke tests automatically stream.** Parallel mode must land a whole 1 GB tar
  before unpacking anything, so with `--limit-per-shard` below 500 the pipeline
  streams instead and stops after a few megabytes. You do not have to think
  about it.

If the server won't serve ranges, it falls back to streaming on its own.

## Performance

Three independent levers, in the order they matter.

### 1. Batched separation (~6×)

Separation is **overhead-bound, not compute-bound**. These models run a fixed
inference window — Kim Vocal 2's is 256 frames × 1024 hop = **5.94s at
44.1 kHz** — and zero-pad anything shorter up to it. Filimo utterances average
about **2 seconds**, so one call per clip pays for ~6s of work per 2s of audio,
plus per-call setup. Measured on the same 19s of audio:

| | throughput |
|---|---|
| one `separate()` call per clip | 1.54× realtime |
| the same audio concatenated | 6.41× realtime |
| a long (114s) file | **9.65× realtime** |

~0.94s of fixed cost per call — **76% of separation time was overhead.**
Profiling ruled out I/O; pydub and ffmpeg together accounted for under 0.5s of
12.3s.

So the pipeline packs clips into ~5-minute inputs, separates once, and slices
the result apart at known offsets. Two details keep that faithful, so a clip's
output never depends on which batch it landed in:

- each clip is **peak-normalised before packing and restored after**, because
  the separator normalises its input globally — otherwise one loud clip would
  quietly attenuate its neighbours;
- a **0.5s silence guard** sits between clips so neighbouring audio cannot
  bleed across a boundary inside the model's receptive field.

That is a claim about fidelity, so it ships with a test that checks it:

```bash
python scripts/validate_batching.py --audio-dir work/raw/<slug>/shard_001
```

It separates the same clips both ways and reports per-clip correlation, log
spectral distance and DNSMOS delta. Measured on filimo shard 1 with Kim Vocal 2:

```
                corr    lsd dB   d_ovrl
mean          0.9995      8.45   +0.151
worst         0.9981     12.00   +0.559
```

Correlation ≥ 0.998 — the same audio. The DNSMOS delta is **positive**, so
packing is slightly *better* than separating clips alone, not merely
equivalent. That is not luck: a 2-second clip separated on its own is
zero-padded to the model's 5.94s window, so the network sees mostly silence,
which is nothing like its training distribution. Inside a packed batch it
always sees a full, realistic window. (LSD sits near 8 dB even at this
correlation because the two runs differ mainly in the low-level residual
noise floor, where tiny absolute differences are huge in dB. Correlation is
the metric that decides.)

If correlation ever drops below ~0.99, raise `separate.batch_guard_seconds`.
Disable packing entirely with `--no-batch-clips`.

### 2. Parallel CPU stages (~N cores)

Prefilter, VAD and quality are embarrassingly parallel and go through a
pluggable backend:

| backend | when |
|---|---|
| `serial` | debugging; no pickling, one process |
| `process` | **default.** Local process pool — right for a single server |
| `ray` | several machines. Actor pool keeps models resident between calls |

```bash
vcprep run --backend process --num-workers 16
vcprep run --backend ray --ray-address auto      # needs: pip install -e '.[ray]'
```

Workers are `spawn`-ed, not forked (torch and onnxruntime do not survive a fork
safely, and a forked CUDA context is a hard error), and each worker pins itself
to one thread so N workers do not oversubscribe the machine. DNSMOS and Silero
load once per worker and amortise over thousands of clips.

**A pool is not free.** Each worker is a fresh interpreter that imports torch
and loads Silero and DNSMOS. On a 12-clip unit a 4-worker pool measured ~55%
*slower* than serial. So `process` falls back to serial below
`backends.MIN_PARALLEL_ITEMS` (256 records) automatically — you do not have to
think about it. Real shards hold thousands of clips per unit, where the pool
wins decisively; small `--limit-per-shard` smoke tests stay fast.

**On Ray specifically:** on one machine it is not faster than `process` — it
adds a head node and an object store for no gain. It earns its keep across
machines, which is why it is implemented behind the same interface but is not
the default. Switching is a flag, not a rewrite.

### Where DNSMOS runs: GPU or CPU

Both are supported and neither is universally right, so it stays your choice.

A typical clip is **one 9-second window at batch size 1**, so DNSMOS is
latency-bound, not compute-bound — a GPU replaces ~41 ms of CPU math with a
kernel launch and two transfers, and sits idle in between. Low GPU utilization
during scoring is the workload's shape, not a misconfiguration.

That makes it a question of how many cores you have. CPU scoring runs at
~24 clips/s per core:

| cores | ~500k clips |
|---|---|
| 2 | ~3 h |
| 8 | ~43 min |
| 16 | ~22 min |
| 32 | ~11 min |

With plenty of cores the CPU pool wins outright and costs no VRAM. With two,
it cannot keep up and the GPU is the better home — even though utilization
stays low, it beats a two-core pool.

```bash
vcprep run --device cuda              # force GPU, whatever the worker count
vcprep run --device cpu               # force CPU, spread over the pool
vcprep run --device auto              # default (see below)
```

`auto` picks the GPU when available **and** workers ≤ `quality.max_gpu_workers`
(default 4), otherwise CPU — because each worker builds its own DNSMOS session,
and on `auto` with CUDA present that is one CUDA context per worker, hundreds of
MB apiece, all contending. Explicit `cuda` or `cpu` is always obeyed.

The same `--device` flag exists on `scripts/compare_separators.py` and
`scripts/validate_batching.py`.

### 3. Separation stays single-process on purpose

It is not routed through the backend. Running several copies would multiply
VRAM rather than throughput; its speedup comes from batching instead.

---

## Design notes

**Prefilter runs before the GPU.** Separation dominates runtime, and a
movie-sourced corpus is full of music- and effects-only clips. A lenient
energy+VAD pass on the raw audio skips those before they cost GPU time. It only
drops the unambiguous; borderline calls go to the post-separation VAD, which
sees a far cleaner signal.

**Silence detection is neural, not a dB threshold.** The dominant failure mode
is not digital silence but clips that were pure music: after Kim Vocal 2 strips
the instrumental, what remains is low-level residue sitting comfortably above
any sane dBFS gate while containing no speech. Silero VAD separates those cases;
RMS cannot. In the smoke test, 6 of 25 clips passed an energy gate and were
correctly caught here as `no_speech`.

**The separator defaults to Mel-Band RoFormer** (`--model-name` takes any model
in audio-separator's registry; `--uvr-model` takes a local checkpoint; the
config carries both MDX-Net and MDXC/RoFormer parameter blocks and uses
whichever matches). **Benchmark it on your own hardware before committing** —
published SDR did not predict what happened here.

Measured on 40 filimo clips, same audio, both models:

| model | SIG | BAK | OVRL | xRT (MPS) |
|---|---|---|---|---|
| Kim Vocal 2 (MDX-Net) | 3.124 | 3.402 | 2.577 | **6.86** |
| Mel-Band RoFormer | 3.090 | **3.540** | 2.602 | 0.58 |

RoFormer's headline advantage is +2.4 dB SDR on studio music benchmarks. On
this corpus that converts to **BAK +0.137** — real, above the noise floor, and
on the axis that matters most here — but **OVRL +0.025**, which is nothing, and
SIG is marginally *worse*: harder background removal costs a little voice. The
limiting factor on lossy, band-limited movie audio is the source material, not
the separator's residual.

The cost ratio above (~12×) is an **Apple Silicon artifact** and should not be
trusted for your server: RoFormer is a transformer and parallelises far better
on CUDA than through MPS. Re-measure on the actual GPU:

```bash
python scripts/compare_separators.py --audio-dir work/raw/<slug>/shard_001 \
    --model vocals_mel_band_roformer.ckpt --model /path/to/Kim_Vocal_2.onnx \
    --limit 300
```

If RoFormer lands within ~2× of Kim Vocal 2 on your GPU, take it for the BAK
gain. If it is still 10× slower, Kim Vocal 2 buys far more corpus per GPU-hour
for a 0.025 OVRL difference.

**Quality uses four scorers, and all of DNSMOS's axes.**

| scorer | contributes | why |
|---|---|---|
| DNSMOS P.835 | SIG / **BAK** / OVRL | BAK measures residual background — a direct report card on the separator |
| NISQA v2 | **discontinuity** | the axis DNSMOS lacks; MDX-Net fails by warbling and chopping |
| SQUIM | predicted PESQ/STOI | free with torchaudio, batches on GPU |
| heuristics | **bandwidth**, clipping, inter-word SNR | structural defects MOS models score right through |

Bandwidth earns its place. In the smoke test one clip scored DNSMOS
`ovrl 3.08 / sig 3.47 / bak 3.86` — excellent by every MOS axis — with a
spectral bandwidth of **6.5 kHz**. It is a lowpassed streaming rip, and training
a 22.05 kHz model on it teaches the vocoder to synthesise a band the data never
contains. Only the bandwidth check catches that.

NISQA and SQUIM are opt-in (`--with-nisqa`, `--with-squim`). NISQA carries an
academic/non-commercial licence — check it before commercial use.

**Failing quality is a demotion, not a deletion.** `low_quality/` keeps the
audio and its scores, so raising or lowering the bar later is a re-split.

---

## ⚠️ Calibrate before the full run

The shipped thresholds are literature defaults for **English** speech that has
not been through source separation. On this corpus they are aggressive — a
25-clip sample kept **31%**:

```
metric            p10     p25     p50     p75     p90
dnsmos_ovrl      2.02    2.15    2.76    2.97    3.03
dnsmos_bak       3.17    3.40    3.52    3.65    3.90
bandwidth_hz     4253    6245    8010    8581    9475
```

Filimo audio is heavily band-limited (median 8 kHz), so `min_bandwidth_hz: 7000`
alone discards roughly a quarter of it. That may well be the right call for a
22.05 kHz model — but it should be *your* call, made against a real
distribution. Run a few hundred clips, then:

```bash
vcprep calibrate --target-keep 0.85
```

It prints percentiles per axis, a suggested threshold for each, and the joint
keep rate your current config would produce. Sample size matters: the numbers
above come from 16 scored clips and are illustrative, not a calibration.

---

## Fine-tuning

```bash
bash scripts/setup_seedvc.sh          # clone + requirements
vcft prepare                          # hardlinks clean/ into a seed-vc dataset dir
vcft train --dry-run                  # print the accelerate command
vcft train --num-processes 4 --mixed-precision bf16
vcft status                           # dataset summary + checkpoints
```

`prepare` **hardlinks** by default, so the training set costs no extra disk. It
enforces seed-vc's 1–30 s window and can apply a stricter quality floor than the
pipeline used (`--min-dnsmos-ovrl`) without re-materialising anything.

V2 has two trainable stages. `--train-cfm` (default) carries acoustic quality
and adapts to a new language relatively cheaply. `--train-ar` needs considerably
more data before it stops overfitting, so it is off by default.

Checkpoints land in `<seed-vc>/runs/<run-name>/`.

---

## Layout

```
vcprep/
  cli.py  config.py  runner.py  audio.py  vad_engine.py  calibrate.py
  manifest.py    Record, Manifest, ManifestStore (per-unit partitions)
  stages.py      per-record logic, shared by the serial and parallel paths
  backends.py    serial | process | ray
  nodes/     fetch  prefilter  separate  vad  quality  materialize
  metrics/   dnsmos  nisqa  squim  heuristics
seedvc_ft/
  cli.py  config.py  prepare.py  train.py
configs/   pipeline.yaml  finetune.yaml  sources.example.yaml
scripts/   setup_server.sh  setup_nisqa.sh  setup_seedvc.sh
           validate_batching.py    packed vs per-clip separation fidelity
           compare_separators.py   separation models by DNSMOS on your audio
```

`stages.py` exists so the serial loop and the pool workers run *exactly* the
same per-record code — there is no second implementation to drift.
