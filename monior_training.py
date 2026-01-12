import time
import argparse
import os
import sys
from datetime import datetime
from plot_dashboard import plot_dashboard, parse_log

def monitor(log_path, interval=60):
    """
    Monitors the training log and updates the dashboard when new epochs are detected.
    """
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return

    print(f"👀 Monitoring: {log_path}")
    print(f"⏱️  Refresh interval: {interval} seconds")
    print("---------------------------------------------------")

    last_epoch = -1
    
    try:
        while True:
            # 1. Parse the log safely
            try:
                data = parse_log(log_path)
                epochs = data['epoch']
            except Exception as e:
                # Log might be incomplete/being written to, just wait
                epochs = []

            # 2. Check for updates
            if epochs:
                current_max_epoch = max(epochs)
                
                if current_max_epoch > last_epoch:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚀 New Epoch Detected: {current_max_epoch}")
                    
                    # 3. Update the Plot
                    plot_dashboard(log_path)
                    
                    last_epoch = current_max_epoch
                else:
                    # Heartbeat dot so you know it's alive
                    sys.stdout.write(".")
                    sys.stdout.flush()
            else:
                sys.stdout.write("w") # Waiting for first epoch
                sys.stdout.flush()

            # 4. Sleep
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str, help="Path to train.log")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    args = parser.parse_args()
    
    monitor(args.log_file, args.interval)
