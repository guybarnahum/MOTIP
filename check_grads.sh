#!/bin/bash

LOG_PATH=${1:-"train.log"}

if [ ! -f "$LOG_PATH" ]; then
    echo "Error: File $LOG_PATH not found."
    exit 1
fi

echo -e "      Step | DETR Norm  | ID Norm    | ID Loss   | Δ Loss % | Status"
echo -e "-----------|------------|------------|-----------|----------|------------"

# Strip color codes and process with awk
sed 's/\x1b\[[0-9;]*m//g' "$LOG_PATH" | awk '
BEGIN { prev_loss = 0 }
/\[[0-9]+\/[0-9]+\]/ && /detr_grad_norm/ {
    # Extract step
    match($0, /\[([0-9]+)\//, step_arr);
    step = step_arr[1];

    # Extract norms
    match($0, /detr_grad_norm = ([0-9.]+)/, detr_arr);
    d_norm = detr_arr[1];
    match($0, /other_grad_norm = ([0-9.]+)/, id_arr);
    o_norm = id_arr[1];

    # Extract ID Loss
    match($0, /id_loss = ([0-9.]+)/, loss_arr);
    curr_loss = loss_arr[1];

    # Calculate Delta
    delta_str = "---";
    if (prev_loss > 0) {
        delta = ((curr_loss - prev_loss) / prev_loss) * 100;
        delta_str = sprintf("%+.2f%%", delta);
    }
    prev_loss = curr_loss;

    # Determine Status
    status = "OK";
    if (o_norm < 0.5) status = "❄️  STALLED";
    else if (o_norm > 25.0) status = "🔥 EXPLODE";
    else if (d_norm > o_norm * 50) status = "⚖️  IMBAL";
    else if (delta < -1.0) status = "📉 SINKING";
    else if (o_norm > 3.0) status = "🚀 ACTIVE";

    printf "%10s | %10.4f | %10.4f | %9.3f | %8s | %s\n", step, d_norm, o_norm, curr_loss, delta_str, status
}'