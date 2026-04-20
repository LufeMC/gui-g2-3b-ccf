# gui-g2-3b-ccf

Inference-time icon refinement for [GUI-G2-3B](https://huggingface.co/inclusionAI/GUI-G2-3B) on the ScreenSpot-v2 benchmark. **+2.2pp on icons, +3.9pp on web, zero extra training.**

The shipped artifact is a small Python wrapper (Cursor-Centric Focusing) around the unmodified base model. It runs the model twice -- once on the full screenshot for a coarse prediction, then once more on a 2x-zoomed crop centered on that prediction -- and maps the refined coordinates back to the original image. The technique is generalized from the [GUI-Cursor paper](https://arxiv.org/abs/2509.21552); the engineering work here is making it production-quality (greedy-only, coarse-pass downsizing, optional type-aware gating).

**[Hugging Face model card with quickstart](https://huggingface.co/luisf-mc/gui-g2-3b-ccf)** | **[Full benchmark writeup with all 9 phases](benchmarks/results.md)**

## Headline numbers

ScreenSpot-v2, 1272 samples, greedy decoding, MAX_PIXELS=12,845,056, H200 GPU with flash-attn 2.

| Configuration | Overall | Desktop | Mobile | Web | Icon | Text |
|---|---|---|---|---|---|---|
| GUI-G2-3B (baseline, no changes) | 89.2% | 91.3% | 88.0% | 84.2% | 80.5% | 96.0% |
| **GUI-G2-3B + CCF** | **88.9%** | 91.3% | 88.0% | **88.1%** | **82.7%** | 93.7% |
| GUI-G2-3B + CCF + type-gate | 88.9% | 91.0% | **89.2%** | 87.0% | 82.7% | 93.7% |

Net effect: +2.2pp on icons (the hardest split), +3.9pp on web, ~tied overall. The text loss is mitigated by the optional type-aware gate (which turns it into +1.2pp on mobile instead).

## Repo layout

```
src/                       Core CCF + cursor-environment + training scripts
  cursor_ccf.py              CCF inference wrapper (the published artifact)
  cursor_ccf_cursor.py       CCF for cursor-style models (research code)
  eval.py                    ScreenSpot-v2 eval script with --ccf, --ccf-type-gate flags
  eval_cursor.py             Eval script for cursor-action-trained models
  cursor_*.py                Multi-step cursor RL infrastructure (Phase 5/6)
  train_bbox_sft.py          Bbox-format SFT trainer (Phase 6/7/8)
  train_cursor_grpo.py       Multi-step GRPO trainer for cursor-style models
  train_cursor_sft.py        SFT warmstart for cursor schema
tests/                     220 unit tests, all green
scripts/                   Data prep + GPU kickoff scripts
benchmarks/results.md      Full per-phase writeup (87 negative results documented honestly)
```

## Quickstart

```bash
git clone https://github.com/LufeMC/gui-g2-3b-ccf.git
cd gui-g2-3b-ccf
pip install -r requirements.txt

# Eval on ScreenSpot-v2 (download the dataset first from HuggingFace)
python src/eval.py \
    --base-model inclusionAI/GUI-G2-3B \
    --data /path/to/screenspot-v2 \
    --ccf --coarse-max-pixels 1500000 \
    --attn-impl flash_attention_2

# Add type-aware gating (optional, +1.2pp on mobile)
python src/eval.py ... --ccf --ccf-type-gate
```

For a self-contained inference example (no eval pipeline needed), see [the predict.py on the HuggingFace model card](https://huggingface.co/luisf-mc/gui-g2-3b-ccf/blob/main/predict.py).

## What's NOT in this repo (intentional)

This repo contains the **research + benchmarking** work. The product side lives elsewhere:

- **Hosted inference API + customer onboarding**: separate private repo
- **Web playground**: separate repo
- **Per-customer LoRA adapter training pipeline**: separate private repo
- **Larger experiments / negative results from Phases 0-2**: not on the published path; documented in `benchmarks/results.md` for completeness but not maintained here

## The 9-phase journey (TL;DR)

We tried 6 different training experiments before concluding that, at 3B + a few-thousand-sample fine-tuning budget, GUI-G2-3B is at or near the achievable optimum on ScreenSpot-v2. The full writeup with per-phase numbers, hardware budgets, and post-mortems is in [`benchmarks/results.md`](benchmarks/results.md). Highlights:

| Phase | Approach | Result |
|---|---|---|
| 4 | CCF inference wrap (no training) | **+2.2pp icon -- the only "ours" win** |
| 5 | Multi-step cursor RL (GUI-Cursor replication) | -15pp |
| 6 | Reward-fixed cursor + bbox SFT pivot | -7pp |
| 7 (probe) | Self-distillation (validates hypothesis) | -1.0pp |
| 8 | 7B teacher distillation, 2k samples | -1.5pp overall, +4.6pp web |
| 9 | CCF + type-aware gate | tied Phase 4 (+1.2pp mobile) |

The single learning that generalizes: **label quality dominates data quantity by an order of magnitude** when fine-tuning a saturated 3B model on a few thousand samples. Self-labels lose 1pp; ground-truth labels lose 7pp; both are dwarfed by inference-time tricks that don't touch the weights.

## Running the tests

```bash
pip install pytest Pillow
python -m pytest tests/ -q
# 220 passed
```

The test suite covers the CCF math, the cursor environment, the reward function, the trainer plumbing, and the eval adapter. No GPU required.

## Citing this work

```bibtex
@misc{guig2_3b_ccf,
  title  = {GUI-G2-3B + CCF: inference-time icon refinement for screen grounding},
  author = {Moncer, Luis F.},
  year   = {2026},
  url    = {https://huggingface.co/luisf-mc/gui-g2-3b-ccf}
}
```

Plus the work this builds on:

```bibtex
@misc{guig2,
  title  = {GUI-G2-3B},
  author = {inclusionAI},
  year   = {2025},
  url    = {https://huggingface.co/inclusionAI/GUI-G2-3B}
}

@misc{guicursor,
  title         = {GUI-Cursor: Cursor-Centric Focusing for GUI Grounding via Multi-Step RL},
  year          = {2025},
  eprint        = {2509.21552},
  archiveprefix = {arXiv}
}
```

## License

Apache 2.0.
