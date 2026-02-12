#!/bin/bash

# --- 1. Argument Parsing & Validation ---
if [ -z "$1" ]; then
    echo "❌ Error: Missing config path."
    echo "Usage: ./train-start.sh <path/to/config.yaml>"
    exit 1
fi

CONFIG_PATH="$1"
# Ensure we are using absolute paths for tmux reliability
ABS_CONFIG_PATH=$(readlink -f "$CONFIG_PATH")

if [ ! -f "$ABS_CONFIG_PATH" ]; then
    echo "❌ Error: Config file not found at: $ABS_CONFIG_PATH"
    exit 1
fi

# Verify Dataset Existence (sanity check)
DATASET_PATH=$(grep "GT_FOLDER" "$ABS_CONFIG_PATH" | head -n 1 | awk -F': ' '{print $2}' | tr -d '"' | tr -d "'" | tr -d " ")
if [ ! -z "$DATASET_PATH" ] && [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset folder not found at: $DATASET_PATH"
    echo "   (Checked GT_FOLDER value in $ABS_CONFIG_PATH)"
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
echo "⚙️  Config:  $ABS_CONFIG_PATH"
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
# FIX: Added 'env' so stdbuf can handle the environment variable assignment
TRAIN_CMD="stdbuf -oL -eL env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --mixed_precision=fp16 --num_processes=1 train.py --config-path $ABS_CONFIG_PATH --exp-name $UNIQUE_EXP_NAME"

# Step B: Post-Processing (Call the separate script)
POST_CMD="./train-post.sh $ABS_CONFIG_PATH $OUTPUT_ROOT $LOG_FILE"

# Step C: Auto-Dashboard (30 min refresh + Final Sweep)
# Logic: While the training process is found in the process list, update every 1800s.
# Once training finishes, run one final update to catch the last epoch metrics.
INTERVAL=1800
DASH_CMD="sleep 15; while pgrep -f '$UNIQUE_EXP_NAME' > /dev/null; do python plot_dashboard.py $LOG_FILE; echo \"Dashboard updated at \$(date). Next update in 30m...\"; sleep $INTERVAL; done; python plot_dashboard.py $LOG_FILE; echo '✅ Training finished. Final dashboard generated.'"

# Combined Command for Pane 0
# We use ';' to ensure POST_CMD runs even if TRAIN_CMD had an error (to see partial results)
FINAL_CMD="$TRAIN_CMD 2>&1 | tee $LOG_FILE; $POST_CMD"

# --- 4. Launch ---
# Create session in background
tmux new-session -d -s "$SESSION_NAME"

# Pane 0: Primary Training and Post-Processing
tmux send-keys -t "$SESSION_NAME" "$FINAL_CMD" C-m

# Pane 1: Split horizontally and run the Dashboard Monitor
tmux split-window -h -t "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "$DASH_CMD" C-m

# Return focus to the training pane
tmux select-pane -t 0

echo "✅ Training launched!"
echo "   To view output: tmux attach -t $SESSION_NAME"