#!/bin/bash

# Check if log path is provided
LOG_PATH=${1:-"train.log"}

if [ ! -f "$LOG_PATH" ]; then
    echo "Error: File $LOG_PATH not found."
    exit 1
fi

echo -e "      Step | DETR Norm  | ID Norm    | Status"
echo -e "-----------|------------|------------|------------"

# 1. Strip color codes
# 2. Extract Step, DETR norm, and ID norm
# 3. Use awk to calculate status and format output
sed 's/\x1b\[[0-9;]*m//g' "$LOG_PATH" | awk '
/\[[0-9]+\/[0-9]+\]/ && /detr_grad_norm/ {
    # Extract step [100/8003] -> 100
    match($0, /\[([0-9]+)\//, step_arr);
    step = step_arr[1];

    # Extract detr_grad_norm
    match($0, /detr_grad_norm = ([0-9.]+)/, detr_arr);
    d_norm = detr_arr[1];

    # Extract other_grad_norm
    match($0, /other_grad_norm = ([0-9.]+)/, id_arr);
    o_norm = id_arr[1];

    # Determine Status
    status = "OK";
    if (o_norm < 0.5) status = "❄️ STALLED";
    else if (o_norm > 25.0) status = "🔥 EXPLODE";
    else if (d_norm > o_norm * 50) status = "⚖️ IMBAL";
    else if (o_norm > 3.0) status = "🚀 ACTIVE";

    printf "%10s | %10.4f | %10.4f | %s\n", step, d_norm, o_norm, status
}'
