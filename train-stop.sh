#!/bin/bash

TARGET_SESSION="$1"

# 1. Get list of active MOTIP sessions
SESSIONS_RAW=$(tmux list-sessions -F "#{session_name}" 2>/dev/null | grep "^motip_")
SESSIONS_ARRAY=($SESSIONS_RAW)
COUNT=${#SESSIONS_ARRAY[@]}

# 2. Logic to determine target
if [ -z "$TARGET_SESSION" ]; then
    if [ "$COUNT" -eq 0 ]; then
        echo "❌ No active MOTIP training sessions found to stop."
        exit 1
    elif [ "$COUNT" -eq 1 ]; then
        TARGET_SESSION="${SESSIONS_ARRAY[0]}"
        echo "🔍 Found single session: '$TARGET_SESSION'."
    else
        echo "📂 Multiple active sessions found:"
        for s in "${SESSIONS_ARRAY[@]}"; do
            echo "   - $s"
        done
        echo ""
        echo "⚠️  For safety, please specify which one to stop:"
        echo "   Usage: $0 <session_name>"
        exit 1
    fi
fi

# 3. Stop Sequence
tmux has-session -t "$TARGET_SESSION" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "🛑 Sending stop signal (Ctrl+C) to '$TARGET_SESSION'..."
    tmux send-keys -t "$TARGET_SESSION" C-c
    
    echo "⏳ Waiting 5 seconds for graceful shutdown..."
    sleep 5
    
    # Check if it's still there, if so, kill it
    tmux has-session -t "$TARGET_SESSION" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "force-killing remaining session..."
        tmux kill-session -t "$TARGET_SESSION"
    fi
    echo "✅ Session '$TARGET_SESSION' stopped."
else
    echo "❌ Session '$TARGET_SESSION' not found."
fi
