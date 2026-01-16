#!/bin/bash

# --- 1. Argument Parsing & Validation ---
if [ -z "$1" ]; then
    echo "❌ Error: Missing config path."
    echo "Usage: ./train-start.sh <path/to/config.yaml>"
    exit 1
fi

CONFIG_PATH="$1"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Error: Config file not found at: $CONFIG_PATH"
    exit 1
fi

# Verify Dataset Existence (sanity check)
DATASET_PATH=$(grep "GT_FOLDER" "$CONFIG_PATH" | head -n 1 | awk -F': ' '{print $2}' | tr -d '"' | tr -d "'" | tr -d " ")
if [ ! -z "$DATASET_PATH" ] && [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset folder not found at: $DATASET_PATH"
    echo "   (Checked GT_FOLDER value in $CONFIG_PATH)"
    exit 1
fi

# --- 2. Setup Session ---
FILENAME=$(basename -- "$CONFIG_PATH")
EXP_BASE_NAME="${FILENAME%.*}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
UNIQUE_EXP_NAME="${EXP_BASE_NAME}_${TIMESTAMP}"
SESSION_NAME="motip_${TIMESTAMP}"

OUTPUT_ROOT="outputs/${UNIQUE_EXP_NAME}"
mkdir -p "$OUTPUT_ROOT"
LOG_FILE="${OUTPUT_ROOT}/train.log"
touch "$LOG_FILE" 

echo "========================================================"
echo "⚙️  Config:  $CONFIG_PATH"
echo "📂 Output:  $OUTPUT_ROOT"
echo "📝 Log:     $LOG_FILE"
echo "🏷️  ExpID:   $UNIQUE_EXP_NAME"
echo "🖥️  Session: $SESSION_NAME"
echo "========================================================"

# Check for existing session
tmux has-session -t "$SESSION_NAME" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "⚠️  Session '$SESSION_NAME' is already running. Attaching..."
    tmux attach -t "$SESSION_NAME"
    exit 0
fi

# --- 3. Construct the Pipeline Command ---
# Step A: Training
TRAIN_CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --mixed_precision=fp16 --num_processes=1 train.py --config-path $CONFIG_PATH --exp-name $UNIQUE_EXP_NAME"

# Step B: Post-Processing (Call the separate script)
POST_CMD="./train-post.sh $CONFIG_PATH $OUTPUT_ROOT $LOG_FILE"

# Combined Command
FINAL_CMD="$TRAIN_CMD 2>&1 | tee $LOG_FILE; $POST_CMD; read -p 'Press Enter to close session...'"

# --- 4. Launch ---
tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "$FINAL_CMD" C-m

echo "✅ Training launched!"
echo "   To view output: tmux attach -t $SESSION_NAME"