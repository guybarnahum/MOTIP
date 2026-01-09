#!/bin/bash

# --- 1. Argument Parsing ---
if [ -z "$1" ]; then
    echo "❌ Error: Missing config path."
    echo "Usage: ./train-start.sh <path/to/config.yaml>"
    exit 1
fi

CONFIG_PATH="$1"

# --- 2. Derive EXP_NAME from filename ---
# Extract filename (e.g., "bdd_mini.yaml")
FILENAME=$(basename -- "$CONFIG_PATH")
# Remove extension (e.g., "bdd_mini")
EXP_NAME="${FILENAME%.*}"

# Create a unique session name (e.g., "motip_bdd_mini")
SESSION_NAME="motip_${EXP_NAME}"
OUTPUT_DIR="output/${EXP_NAME}"
LOG_FILE="${OUTPUT_DIR}/train.log"

# Create output dir if it doesn't exist (for the log file)
mkdir -p "$OUTPUT_DIR"

echo "⚙️  Config:  $CONFIG_PATH"
echo "🏷️  Exp Name: $EXP_NAME"
echo "🖥️  Session:  $SESSION_NAME"

# --- 3. Check for existing session ---
tmux has-session -t "$SESSION_NAME" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "⚠️  Session '$SESSION_NAME' is already running."
    echo "   Attaching to it now... (Press Ctrl+b, d to detach)"
    sleep 1
    tmux attach -t "$SESSION_NAME"
    exit 0
fi

# --- 4. Build Command (Auto-Resume Logic) ---
# Added PYTORCH_CUDA_ALLOC_CONF before the command
BASE_CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --mixed_precision=fp16 --num_processes=1 train.py --config-path $CONFIG_PATH --exp-name $EXP_NAME"
FINAL_CMD=""

if [ -f "${OUTPUT_DIR}/checkpoint.pth" ]; then
    echo "🔄 Found checkpoint. Resuming..."
    FINAL_CMD="$BASE_CMD --resume ${OUTPUT_DIR}/checkpoint.pth 2>&1 | tee -a $LOG_FILE"
else
    echo "✨ Starting fresh training..."
    FINAL_CMD="$BASE_CMD 2>&1 | tee $LOG_FILE"
fi

# --- 5. Launch in Tmux ---
tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "$FINAL_CMD" C-m

echo "✅ Training launched in session '$SESSION_NAME'."
echo "   Log: $LOG_FILE"
echo "   Run: ./train-attach.sh $SESSION_NAME"
