#!/bin/bash

SESSION_NAME="ec2_test"
LOG_FILE="test_output.log"

# 1. Start a detached tmux session (if not already running)
tmux has-session -t "$SESSION_NAME" 2>/dev/null
if [ $? -ne 0 ]; then
    tmux new-session -d -s "$SESSION_NAME"
    
    # 2. Send the command
    # We use 'unbuffer' or standard redirection. 
    # The sleep 0.1 inside the loop ensures it doesn't spam CPU.
    tmux send-keys -t "$SESSION_NAME" "while true; do echo '✅ Environment is healthy: $(date)'; sleep 5; done > $LOG_FILE" C-m
    
    echo "🚀 Test launched in background session '$SESSION_NAME'."
else
    echo "⚠️  Test session already running."
fi

# 3. Wait for the log file to appear (Fixes the race condition)
echo "⏳ Waiting for log file to initialize..."
while [ ! -f "$LOG_FILE" ]; do
  sleep 0.5
done

echo "📄 Watching log at $LOG_FILE (Press Ctrl+C to stop watching, test will keep running)"
echo "-----------------------------------------------------------------------"
tail -f "$LOG_FILE"
