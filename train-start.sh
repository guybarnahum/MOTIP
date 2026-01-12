#!/bin/bash

# --- 1. Argument Parsing ---
if [ -z "$1" ]; then
    echo "❌ Error: Missing config path."
    echo "Usage: ./train-start.sh <path/to/config.yaml>"
    exit 1
fi

CONFIG_PATH="$1"

# --- 2. Generate Unique Session Info ---
FILENAME=$(basename -- "$CONFIG_PATH")
EXP_BASE_NAME="${FILENAME%.*}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
UNIQUE_EXP_NAME="${EXP_BASE_NAME}_${TIMESTAMP}"

# Pre-create folder & Empty Log File (Crucial for the monitor to start safely)
OUTPUT_ROOT="outputs/${UNIQUE_EXP_NAME}"
mkdir -p "$OUTPUT_ROOT"
LOG_FILE="${OUTPUT_ROOT}/train.log"
touch "$LOG_FILE" 

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

# --- 4. Build The Command ---
# A. Training Command
TRAIN_CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --mixed_precision=fp16 --num_processes=1 train.py --config-path $CONFIG_PATH --exp-name $UNIQUE_EXP_NAME"

# B. Monitor Command (Updates dashboard.png every 60s)
#    We use 'python' assuming venv is active, or use sys.executable from training if needed.
MONITOR_CMD="python monitor_training.py $LOG_FILE --interval 60"

# C. Compound Lifecycle Command
#    1. Start Monitor in Background (&)
#    2. Save its PID ($!)
#    3. Run Training (Blocking)
#    4. Kill Monitor when Training finishes (Success or Fail)
FINAL_CMD="$MONITOR_CMD > /dev/null 2>&1 & MON_PID=\$!; $TRAIN_CMD 2>&1 | tee $LOG_FILE; kill \$MON_PID"

# --- 5. Launch in Tmux ---
tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "$FINAL_CMD" C-m

echo "✅ Training launched with Live Dashboard!"
echo "   To view output: tmux attach -t $SESSION_NAME"