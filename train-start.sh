#!/bin/bash

# --- 1. Argument Parsing ---
if [ -z "$1" ]; then
    echo "❌ Error: Missing config path."
    echo "Usage: ./train-start.sh <path/to/config.yaml>"
    exit 1
fi

CONFIG_PATH="$1"

# --- 2. Generate Unique Session Info ---
# Get filename (e.g. "pretrain_r50_deformable_detr_bdd_mini")
FILENAME=$(basename -- "$CONFIG_PATH")
EXP_BASE_NAME="${FILENAME%.*}"

# Generate Timestamp (e.g. 20260111_093000)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create Unique Experiment Name
# This triggers the code to save to: ./outputs/pretrain_..._20260111_093000/
UNIQUE_EXP_NAME="${EXP_BASE_NAME}_${TIMESTAMP}"

# Pre-create the folder so we can put the log file there immediately
OUTPUT_ROOT="outputs/${UNIQUE_EXP_NAME}"
mkdir -p "$OUTPUT_ROOT"
LOG_FILE="${OUTPUT_ROOT}/train.log"

SESSION_NAME="motip_${TIMESTAMP}"

echo "========================================================"
echo "⚙️  Config:  $CONFIG_PATH"
echo "📂 Output:  $OUTPUT_ROOT"
echo "📝 Log:     $LOG_FILE"
echo "🏷️  ExpID:   $UNIQUE_EXP_NAME"
echo "🖥️  Session: $SESSION_NAME"
echo "========================================================"

# --- 3. Check for existing session ---
tmux has-session -t "$SESSION_NAME" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "⚠️  Session '$SESSION_NAME' is already running."
    echo "   Attaching to it now... (Press Ctrl+b, d to detach)"
    sleep 1
    tmux attach -t "$SESSION_NAME"
    exit 0
fi

# --- 4. Build Command ---
# We ONLY pass --exp-name. We do NOT pass --outputs-dir to avoid the strict config check error.
BASE_CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --mixed_precision=fp16 --num_processes=1 train.py --config-path $CONFIG_PATH --exp-name $UNIQUE_EXP_NAME"

# --- 5. Launch in Tmux ---
tmux new-session -d -s "$SESSION_NAME"
# Pipe output to tee so it goes to the file AND the screen
tmux send-keys -t "$SESSION_NAME" "$BASE_CMD 2>&1 | tee $LOG_FILE" C-m

echo "✅ Training launched!"
echo "   To view output: tmux attach -t $SESSION_NAME"