#!/usr/bin/env bash
# Phase 5 one-shot kickoff script.
#
# Run on a freshly-spun H100/H200 pod. Assumes:
#   - /workspace is the persistent network FS with our existing data
#     (models/gui-g2-3b, data/icon_mining/hard_icons.jsonl, data/screenspot-v2)
#   - /dev/shm has at least 30GB free (we cache models + data there to
#     bypass the slow MooseFS network FS)
#   - The repo's src/ + scripts/ have been scp'd into /workspace/
#
# Usage:
#   bash scripts/run_phase5_gpu.sh setup
#   bash scripts/run_phase5_gpu.sh data
#   bash scripts/run_phase5_gpu.sh sft_vanilla
#   bash scripts/run_phase5_gpu.sh sft_guig2
#   bash scripts/run_phase5_gpu.sh grpo_vanilla
#   bash scripts/run_phase5_gpu.sh grpo_guig2
#   bash scripts/run_phase5_gpu.sh eval CKPT_DIR
#
# Each step writes its log to /workspace/logs/<step>.log and is idempotent:
# rerunning skips already-done work.

set -e
set -o pipefail

mkdir -p /workspace/logs
mkdir -p /workspace/checkpoints

step="${1:-help}"

setup() {
    echo "=== setup: install Python deps + cache model + data to /dev/shm ==="
    pip install transformers==4.49.0 peft==0.13.2 qwen-vl-utils Pillow tqdm 2>&1 | tail -3
    pip install flash-attn==2.7.0.post2 --no-build-isolation 2>&1 | tail -3
    python3 -c "import flash_attn; print('flash-attn OK', flash_attn.__version__)"

    mkdir -p /dev/shm/models /dev/shm/data
    if [ ! -d /dev/shm/models/gui-g2-3b ]; then
        echo "Caching gui-g2-3b to /dev/shm..."
        time cp -r /workspace/models/gui-g2-3b /dev/shm/models/
    fi
    if [ ! -d /dev/shm/models/qwen2.5-vl-3b ]; then
        if [ -d /workspace/models/qwen2.5-vl-3b ]; then
            echo "Caching vanilla Qwen2.5-VL-3B to /dev/shm..."
            time cp -r /workspace/models/qwen2.5-vl-3b /dev/shm/models/
        else
            echo "Downloading Qwen2.5-VL-3B-Instruct from HF..."
            python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-VL-3B-Instruct',
    local_dir='/dev/shm/models/qwen2.5-vl-3b',
    local_dir_use_symlinks=False,
)"
        fi
    fi
    if [ ! -d /dev/shm/data/screenspot-v2 ]; then
        echo "Caching screenspot-v2 to /dev/shm..."
        time cp -r /workspace/data/screenspot-v2 /dev/shm/data/
    fi
    if [ ! -d /dev/shm/data/icon_mining ]; then
        echo "Caching icon_mining to /dev/shm..."
        time cp -r /workspace/data/icon_mining /dev/shm/data/
    fi
    echo "Setup complete."
}

data() {
    echo "=== data: build cursor_train.jsonl + cursor_val.jsonl ==="
    # Extract Playwright subset tar if present and not already extracted.
    if [ -f /workspace/data/cleaned_subset.tar ] && [ ! -d /workspace/data/cleaned_subset ]; then
        echo "Extracting cleaned_subset.tar..."
        cd /workspace/data && tar -xf cleaned_subset.tar && cd /workspace
    fi
    local playwright_dir="/workspace/data/cleaned_subset"
    if [ ! -d "$playwright_dir" ]; then
        # Fall back to full cleaned dir if it exists
        if [ -d /workspace/data/cleaned ]; then
            playwright_dir="/workspace/data/cleaned"
        else
            echo "WARNING: No Playwright data found. Running icons-only."
            playwright_dir=""
        fi
    fi

    cd /workspace
    if [ -n "$playwright_dir" ]; then
        python3 scripts/prep_cursor_train.py \
            --icons /dev/shm/data/icon_mining/hard_icons.jsonl \
            --playwright-dir "$playwright_dir" \
            --samples-per-page 1 \
            --out-train /workspace/data/cursor_train.jsonl \
            --out-val /workspace/data/cursor_val.jsonl
        # Rewrite icon paths so training reads from /dev/shm (fast) instead
        # of /workspace (the slow MooseFS network FS that cost us hours
        # during the CCF eval).
        python3 - <<'PY'
import json
for fname in ('/workspace/data/cursor_train.jsonl', '/workspace/data/cursor_val.jsonl'):
    out = []
    with open(fname) as f:
        for line in f:
            r = json.loads(line)
            r['img_path'] = r['img_path'].replace(
                '/workspace/data/icon_mining', '/dev/shm/data/icon_mining'
            )
            out.append(r)
    with open(fname, 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print(f"Rewrote icon paths in {fname} ({len(out)} records)")
PY
    else
        # Icons-only fallback
        python3 - <<'PY'
import sys, random
sys.path.insert(0, '/workspace/scripts')
from prep_cursor_train import load_icons, stratified_split, write_jsonl
records = load_icons('/dev/shm/data/icon_mining/hard_icons.jsonl')
rng = random.Random(42)
train, val = stratified_split(records, val_size=50, rng=rng)
write_jsonl('/workspace/data/cursor_train.jsonl', train)
write_jsonl('/workspace/data/cursor_val.jsonl', val)
print(f"Wrote {len(train)} train + {len(val)} val (icons only)")
PY
    fi
}

sft_vanilla() {
    echo "=== sft_vanilla: SFT warmstart on Qwen2.5-VL-3B-Instruct ==="
    cd /workspace
    nohup python3 -u src/train_cursor_sft.py \
        --base-model /dev/shm/models/qwen2.5-vl-3b \
        --data /workspace/data/cursor_train.jsonl \
        --output /workspace/checkpoints/sft-vanilla \
        --epochs 1 \
        --lr 1e-5 \
        --lora-r 64 \
        --max-pixels 1003520 \
        --attn-impl flash_attention_2 \
        --log-every 25 \
        > /workspace/logs/sft_vanilla.log 2>&1 &
    echo "PID=$! (tail /workspace/logs/sft_vanilla.log to monitor)"
}

sft_guig2() {
    echo "=== sft_guig2: SFT warmstart on GUI-G2-3B ==="
    cd /workspace
    nohup python3 -u src/train_cursor_sft.py \
        --base-model /dev/shm/models/gui-g2-3b \
        --data /workspace/data/cursor_train.jsonl \
        --output /workspace/checkpoints/sft-guig2 \
        --epochs 1 \
        --lr 1e-5 \
        --lora-r 64 \
        --max-pixels 1003520 \
        --attn-impl flash_attention_2 \
        --log-every 25 \
        > /workspace/logs/sft_guig2.log 2>&1 &
    echo "PID=$! (tail /workspace/logs/sft_guig2.log to monitor)"
}

grpo_vanilla() {
    echo "=== grpo_vanilla: GRPO Run A on vanilla Qwen + SFT warmstart ==="
    cd /workspace
    nohup python3 -u src/train_cursor_grpo.py \
        --policy vlm \
        --base-model /dev/shm/models/qwen2.5-vl-3b \
        --warmstart-adapter /workspace/checkpoints/sft-vanilla/final \
        --data /workspace/data/cursor_train.jsonl \
        --val-data /workspace/data/cursor_val.jsonl \
        --num-steps 250 \
        --save-every 25 \
        --eval-every 25 \
        --prompts-per-step 4 \
        --trajectories-per-prompt 6 \
        --max-steps-per-trajectory 4 \
        --lr 1e-6 \
        --output-dir /workspace/checkpoints/grpo-vanilla \
        --max-pixels 1003520 \
        --attn-impl flash_attention_2 \
        --temperature 1.0 \
        --wandb-project gui-cursor \
        --wandb-run-name grpo-vanilla \
        > /workspace/logs/grpo_vanilla.log 2>&1 &
    echo "PID=$! (tail /workspace/logs/grpo_vanilla.log)"
}

grpo_guig2() {
    echo "=== grpo_guig2: GRPO Run B on GUI-G2 + SFT warmstart ==="
    # Reduced from plan defaults to fit hardware budget. The first guig2
    # run was 4.3 min/step (vs vanilla's 2.0) which would have ETA'd 18h.
    # Cutting prompts*trajs from 24 to 12 and num_steps from 250 to 150
    # brings ETA to ~5h while keeping the GRPO signal. Eval cadence stays
    # at 25 steps so we still get 6 mid-training data points.
    cd /workspace
    nohup python3 -u src/train_cursor_grpo.py \
        --policy vlm \
        --base-model /dev/shm/models/gui-g2-3b \
        --warmstart-adapter /workspace/checkpoints/sft-guig2/final \
        --data /workspace/data/cursor_train.jsonl \
        --val-data /workspace/data/cursor_val.jsonl \
        --num-steps 150 \
        --save-every 25 \
        --eval-every 25 \
        --prompts-per-step 3 \
        --trajectories-per-prompt 4 \
        --max-steps-per-trajectory 4 \
        --lr 1e-6 \
        --output-dir /workspace/checkpoints/grpo-guig2 \
        --max-pixels 1003520 \
        --attn-impl flash_attention_2 \
        --temperature 1.0 \
        --wandb-project gui-cursor \
        --wandb-run-name grpo-guig2 \
        > /workspace/logs/grpo_guig2.log 2>&1 &
    echo "PID=$! (tail /workspace/logs/grpo_guig2.log)"
}

eval_ckpt() {
    local ckpt="$2"
    if [ -z "$ckpt" ]; then
        echo "Usage: $0 eval CKPT_DIR"
        exit 1
    fi
    echo "=== eval_ckpt: ScreenSpot-v2 + CCF for $ckpt ==="
    cd /workspace
    local out_log="/workspace/logs/eval_$(basename $ckpt).log"
    nohup python3 -u src/eval.py \
        --base-model /dev/shm/models/gui-g2-3b \
        --ckpt "$ckpt" \
        --data /dev/shm/data/screenspot-v2 \
        --ccf --zoom-factor 2.0 --coarse-max-pixels 1500000 \
        --attn-impl flash_attention_2 \
        --results-out "/workspace/eval_$(basename $ckpt).json" \
        > "$out_log" 2>&1 &
    echo "PID=$! (tail $out_log)"
}

case "$step" in
    setup) setup ;;
    data) data ;;
    sft_vanilla) sft_vanilla ;;
    sft_guig2) sft_guig2 ;;
    grpo_vanilla) grpo_vanilla ;;
    grpo_guig2) grpo_guig2 ;;
    eval) eval_ckpt "$@" ;;
    help|*)
        cat <<EOF
Usage: $0 <step> [args]

Steps (run in order):
  setup          install deps, cache models + data to /dev/shm
  data           build cursor_train.jsonl + cursor_val.jsonl
  sft_vanilla    SFT warmstart Qwen2.5-VL-3B-Instruct (~1h)
  sft_guig2      SFT warmstart GUI-G2-3B (~1h)
  grpo_vanilla   GRPO 250 steps from vanilla SFT (~3-4h)
  grpo_guig2     GRPO 250 steps from GUI-G2 SFT (~3-4h)
  eval CKPT_DIR  ScreenSpot-v2 + CCF on a single checkpoint (~3.5h)
EOF
        ;;
esac
