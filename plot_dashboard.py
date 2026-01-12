import re
import matplotlib.pyplot as plt
import argparse
import os

def parse_log(log_path):
    data = {
        'epoch': [],
        'loss': [], 'detr_loss': [], 'id_loss': [],
        'class_error': [], 'grad_norm': [],
        'HOTA': [], 'MOTA': [], 'IDF1': []
    }
    
    # UPDATED REGEX: 
    # 1. Added [\-\d\.]+ to capture negative signs (e.g., -0.0000)
    # 2. Made grad_norm capture more robust
    train_pattern = (
        r"\[Finish epoch: (\d+)\].*?"
        r"loss = ([\-\d\.]+);.*?"
        r"detr_loss = ([\-\d\.]+);.*?"
        r"id_loss = ([\-\d\.]+);.*?"
        r"class_error = ([\-\d\.]+);.*?"
        r"detr_grad_norm = ([naninf\-\d\.]+);"
    )
    
    eval_pattern = r"\[Eval epoch: (\d+)\].*?HOTA = ([\-\d\.]+);.*?MOTA = ([\-\d\.]+);.*?IDF1 = ([\-\d\.]+);"

    print(f"Parsing log file: {log_path}...")
    
    with open(log_path, 'r') as f:
        content = f.read()
        
        # 1. Parse Training Data
        matches = list(re.finditer(train_pattern, content))
        print(f"Found {len(matches)} completed training epochs.")
        
        for match in matches:
            data['epoch'].append(int(match.group(1)))
            data['loss'].append(float(match.group(2)))
            data['detr_loss'].append(float(match.group(3)))
            data['id_loss'].append(float(match.group(4)))
            data['class_error'].append(float(match.group(5)))
            
            # Handle 'inf' or 'nan' safely
            grad = match.group(6).lower()
            if 'inf' in grad or 'nan' in grad:
                data['grad_norm'].append(None) 
            else:
                data['grad_norm'].append(float(grad))

        # 2. Parse Eval Data
        eval_map = {}
        for match in re.finditer(eval_pattern, content):
            epoch = int(match.group(1))
            eval_map[epoch] = {
                'HOTA': float(match.group(2)),
                'MOTA': float(match.group(3)),
                'IDF1': float(match.group(4))
            }
        
        # Align eval data to training epochs
        for e in data['epoch']:
            if e in eval_map:
                data['HOTA'].append(eval_map[e]['HOTA'])
                data['MOTA'].append(eval_map[e]['MOTA'])
                data['IDF1'].append(eval_map[e]['IDF1'])
            else:
                # If eval hasn't run for this epoch yet, pad with None or 0
                data['HOTA'].append(None)
                data['MOTA'].append(None)
                data['IDF1'].append(None)

    return data

def plot_dashboard(log_path):
    data = parse_log(log_path)
    epochs = data['epoch']
    
    if not epochs:
        print("❌ No completed epochs found in log! (Check regex or log format)")
        return

    # Create a 2x2 Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Dashboard: {os.path.basename(os.path.dirname(log_path))}\n(Epochs 0-{max(epochs)})', fontsize=16)

    # Plot 1: Main Losses
    axs[0, 0].plot(epochs, data['loss'], 'r-o', label='Total Loss')
    axs[0, 0].plot(epochs, data['detr_loss'], 'g--', label='DETR Loss')
    axs[0, 0].plot(epochs, data['id_loss'], 'b:', label='ID Loss')
    axs[0, 0].set_title('Loss Components (Lower is Better)')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation Metrics
    # Filter Nones before plotting to avoid errors
    valid_idxs = [i for i, x in enumerate(data['HOTA']) if x is not None]
    if valid_idxs:
        valid_epochs = [epochs[i] for i in valid_idxs]
        axs[0, 1].plot(valid_epochs, [data['HOTA'][i] for i in valid_idxs], 'b-s', linewidth=2, label='HOTA')
        axs[0, 1].plot(valid_epochs, [data['MOTA'][i] for i in valid_idxs], 'g--', label='MOTA')
        axs[0, 1].plot(valid_epochs, [data['IDF1'][i] for i in valid_idxs], 'm:', label='IDF1')
    axs[0, 1].set_title('Validation Performance (Higher is Better)')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Score (%)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Plot 3: Stability (Gradient Norm)
    valid_grads = [(e, g) for e, g in zip(epochs, data['grad_norm']) if g is not None]
    if valid_grads:
        axs[1, 0].plot(*zip(*valid_grads), 'k-x', linewidth=1, label='Grad Norm')
        axs[1, 0].set_title('Training Stability (Gradient Norm)')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Norm')
        axs[1, 0].grid(True, alpha=0.3)
    else:
        axs[1, 0].text(0.5, 0.5, 'Gradients were Inf/NaN', ha='center', transform=axs[1, 0].transAxes)

    # Plot 4: Classifier Accuracy
    axs[1, 1].plot(epochs, data['class_error'], 'r-d', label='Class Error')
    axs[1, 1].set_title('Classification Error (Lower is Better)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Error Rate')
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = os.path.dirname(log_path)
    output_file = os.path.join(output_dir, "dashboard.png")

    plt.savefig(output_file)
    print(f"✅ Dashboard saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str, help="Path to train.log")
    args = parser.parse_args()
    plot_dashboard(args.log_file)