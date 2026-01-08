#!/bin/bash

TARGET_SESSION="$1"

# 1. Get list of active MOTIP sessions
SESSIONS_RAW=$(tmux list-sessions -F "#{session_name}" 2>/dev/null | grep "^motip_")
SESSIONS_ARRAY=($SESSIONS_RAW)
COUNT=${#SESSIONS_ARRAY[@]}

# 2. Logic to determine target
if [ -z "$TARGET_SESSION" ]; then
    if [ "$COUNT" -eq 0 ]; then
        echo "❌ No active MOTIP training sessions found."
        exit 1
    elif [ "$COUNT" -eq 1 ]; then
        TARGET_SESSION="${SESSIONS_ARRAY[0]}"
        echo "🔍 Found single session: '$TARGET_SESSION'. Attaching..."
    else
        echo "📂 Multiple active sessions found:"
        for s in "${SESSIONS_ARRAY[@]}"; do
            echo "   - $s"
        done
        echo ""
        echo "Usage: $0 <session_name>"
        exit 1
    fi
fi

# 3. Attach
tmux has-session -t "$TARGET_SESSION" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "🚀 Attaching to '$TARGET_SESSION'..."
    echo "   (Press Ctrl+b, then d to detach)"
    sleep 1
    tmux attach -t "$TARGET_SESSION"
else
    echo "❌ Session '$TARGET_SESSION' not found."
fi
