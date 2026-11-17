# Learning-rate schedule experiment

## Objective

Track B: reach the frozen next-token validation CE target of
`3.0081611946225166` using fewer than `2,999,910,400` predicted
training-token exposures.

The first experiment changes the peak learning rate while keeping the
architecture, tokenizer, prepared data, data order, evaluation protocol,
batch size, seed, and WSD schedule unchanged.

## Reference configuration

- Model: MiniPLE PLE64, 146,496,016 parameters
- Seed: 1337
- AdamW peak LR: `6e-4`
- Muon peak LR: `8e-4`
- Warmup: `10,000,000` requested tokens
- Stable stage: `2,100,000,000` requested tokens
- Decay stage: `900,000,000` requested tokens
- Evaluation: 128 sequences, 2048 tokens, every 250 updates

All runs must use a **single reconstructed training corpus**, not the historical
training slices. The historical validation threshold remains the official
Track B target, but the changed model must also be compared with a fresh
unchanged control.

## First comparison

| Run | ADAMW_LR | MUON_LR | Main change |
| --- | --- | --- | --- |
| baseline | 6e-4 | 8e-4 | None |
| lr_adamw_low | 5e-4 | 8e-4 | AdamW peak learning rate |
| lr_adamw_high | 7e-4 | 8e-4 | AdamW peak learning rate |

These are **candidate settings**, not measured improvements. Test Muon LR in
a separate follow-up comparison to isolate its effect.

## Setup (run in a compatible Linux/CUDA environment)

First follow `docs/SETUP.md`, including installation, reference verification,
training-data preparation, and `--validate-data-only`. The data must be
prepared once, validated, and held fixed for the comparisons.

```bash
python3 scripts/check_starter.py
python3 scripts/verify_artifacts.py --checkpoints none
python -m unittest discover -s tests -v
python -u prepare_data.py --out-dir data/ultrafineweb_3b
python train.py --data-dir data/ultrafineweb_3b --validate-data-only
```

Then run each configuration with a **new** output directory:

```bash
GPU_ID=0 OUT_DIR=runs/lr_baseline \
  bash scripts/train.sh

GPU_ID=0 OUT_DIR=runs/lr_adamw_low ADAMW_LR=5e-4 \
  bash scripts/train.sh

GPU_ID=0 OUT_DIR=runs/lr_adamw_high ADAMW_LR=7e-4 \
  bash scripts/train.sh
```

Do not run all three until the environment is verified and GPU time is
available. The default launcher executes the full 3B-token recipe; it is
**not** a smoke test. A shorter pilot requires an explicit, matched budget
and correctly prepared training data.

## Evaluation and reporting

Use the frozen full-vocabulary next-token CE. The first *scheduled*
validation meeting the target is the qualifying Track B result. Do not
interpolate between evaluations. Record:

- code revision, seed, data hashes, effective schedule, and command;
- validation CE versus consumed tokens for every run;
- first qualifying scheduled validation and its preceding validation;
- checkpoint SHA256, GPU, memory use, wall time, and any repeated exposures;
- full zero-shot benchmark results for final candidates.

**Important:** the stock trainer only saves periodic checkpoints every 1000
updates. Before any scored Track B training, implement and test saving a
checkpoint immediately upon a qualifying scheduled validation. Do not claim
a qualifying checkpoint exists until it has been saved and verified.

The baseline and candidates have **not** been trained yet.
