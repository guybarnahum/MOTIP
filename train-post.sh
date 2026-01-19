#!/bin/bash
# Usage: ./train-post.sh <CONFIG_PATH> <OUTPUT_ROOT> <LOG_FILE>

if [ "$#" -ne 3 ]; then
    echo "Usage: ./train-post.sh <CONFIG_PATH> <OUTPUT_ROOT> <LOG_FILE>"
    exit 1
fi

CONFIG_PATH="$1"
OUTPUT_ROOT="$2"
LOG_FILE="$3"

echo "========================================================"
echo "📊 Starting Post-Training Analysis"
echo "========================================================"

# --- 1. Generate Dashboard (Static Plot) ---
echo "📈 Generating Training Dashboard..."

# ⚠️ CHANGED: Calling plot_dashboard.py directly as requested
if [ -f "plot_dashboard.py" ]; then
    python plot_dashboard.py "$LOG_FILE"
    
    # Check if image was created (assumes it saves to the output dir or current dir)
    if [ -f "${OUTPUT_ROOT}/train.png" ] || [ -f "train.png" ]; then
        echo "✅ Dashboard generated."
        # If it saved to CWD, move it to output (cleanup)
        if [ -f "train.png" ] && [ ! -f "${OUTPUT_ROOT}/train.png" ]; then
            mv train.png "$OUTPUT_ROOT/"
        fi
    else
        echo "⚠️  Dashboard image not found (check where plot_dashboard.py saves it)."
    fi
else
    echo "❌ Error: plot_dashboard.py not found in current directory."
fi

# --- 2. Find Best Checkpoint (Smart Fallback) ---
echo "🔍 Searching for checkpoints..."

# Priority 1: Best Metric Checkpoint (e.g. checkpoint_best_idf1.pth)
CKPT=$(find "$OUTPUT_ROOT" -name 'checkpoint_best_*.pth' -print -quit)

# Priority 2: Standard Latest Checkpoint (checkpoint.pth)
if [ -z "$CKPT" ]; then 
    CKPT=$(find "$OUTPUT_ROOT" -name 'checkpoint.pth' -print -quit)
fi

# Priority 3: Highest Numbered Epoch (checkpoint_8.pth, checkpoint_9.pth...)
# This is crucial for your current output structure
if [ -z "$CKPT" ]; then
    # ls -v sorts naturally (1, 2... 9, 10), tail -n 1 gets the last one
    CKPT=$(ls -v "${OUTPUT_ROOT}/checkpoint_"*.pth 2>/dev/null | tail -n 1)
fi

if [ -z "$CKPT" ]; then
    echo "❌ No checkpoint found in $OUTPUT_ROOT. Skipping visualization."
    exit 0
fi
echo "✅ Found checkpoint: $CKPT"

# --- 3. Extract Dataset Path from Config ---
# We verify where to run the visualization test
VAL_DATASET=$(grep "GT_FOLDER" "$CONFIG_PATH" | head -n 1 | awk -F': ' '{print $2}' | tr -d '"' | tr -d "'" | tr -d " ")

if [ -z "$VAL_DATASET" ]; then
    echo "⚠️  Could not auto-detect GT_FOLDER from config. Defaulting to ./datasets/DanceTrack/val"
    VAL_DATASET="./datasets/DanceTrack/val"
fi

# --- 4. Run Visualization ---
echo "🎥 Running Visualization on: $VAL_DATASET"
VIZ_OUT="${OUTPUT_ROOT}/viz_val"
mkdir -p "$VIZ_OUT"

python viz_val.py \
  --config "$CONFIG_PATH" \
  --checkpoint "$CKPT" \
  --dataset_root "$VAL_DATASET" \
  --output_dir "$VIZ_OUT" \
  --score_thresh 0.65

echo "========================================================"
echo "✅ Analysis Complete!"
echo "   - Log:       $LOG_FILE"
echo "   - Dashboard: ${OUTPUT_ROOT}/train.png"
echo "   - Videos:    $VIZ_OUT"
echo "========================================================"