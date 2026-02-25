#!/bin/bash
# check_grads_v3.sh - The Ultimate MOTIP Dashboard

LOG_PATH=${1:-"train.log"}
[ ! -f "$LOG_PATH" ] && echo "Error: File $LOG_PATH not found." && exit 1

echo -e "      Step | DETR_N | ID_N   | ID_Loss | Δ_ID_% | Card_Err | Cls_Err | Status"
echo -e "-----------|--------|--------|---------|--------|----------|---------|------------"

sed 's/\x1b\[[0-9;]*m//g' "$LOG_PATH" | awk '
BEGIN { prev_loss = 0 }
/\[[0-9]+\/[0-9]+\]/ && /detr_grad_norm/ {
    match($0, /\[([0-9]+)\//, s); step = s[1];
    match($0, /detr_grad_norm = ([0-9.]+)/, d); dn = d[1];
    match($0, /other_grad_norm = ([0-9.]+)/, i); inr = i[1];
    match($0, /id_loss = ([0-9.]+)/, l); cl = l[1];
    match($0, /cardinality_error = ([0-9.]+)/, ce); card = ce[1];
    match($0, /class_error = ([0-9.]+)/, cls); clse = cls[1];

    ds = "---";
    if (prev_loss > 0) {
        delta = ((cl - prev_loss) / prev_loss) * 100;
        ds = sprintf("%+5.1f%%", delta);
    }
    prev_loss = cl;

    st = "OK";
    if (inr < 0.5) st = "❄️ STALL";
    else if (delta < -5.0) st = "📉 SINKING";
    else if (delta > 20.0) st = "🛰️ NEW_DATA"; # High loss spike from new density
    else if (inr > 3.0) st = "🚀 ACTIVE";

    printf "%10s | %6.1f | %6.1f | %7.2f | %6s | %8.1f | %7.3f | %s\n", step, dn, inr, cl, ds, card, clse, st
}'