import re
import matplotlib.pyplot as plt
import argparse
import os
import math

def parse_log(log_path):
    data = {
        'epoch': [],
        'loss': [], 'detr_loss': [], 'id_loss': [],
        'class_error': [], 'grad_norm': [],
        'HOTA': [], 'MOTA': [], 'IDF1': []
    }
    
    # Regex for Training Summary (Finish epoch line)
    # Matches: [Finish epoch: 0] ... loss = 22.03 ...
    train_pattern = r"\[Finish epoch: (\d+)\].*?loss = ([\d\.]+);.*?detr_loss = ([\d\.]+);.*?id_loss = ([\d\.]+);.*?class_error = ([\d\.]+);.*?detr_grad_norm = ([nan\d\.]+inf|[nan\d\.]+);"
    
    # Regex for Evaluation Summary
    # Matches: [Eval epoch: 0] ... HOTA = 0.00 ...
    eval_pattern = r"\[Eval epoch: (\d+)\].*?HOTA = ([\d\.]+);.*?MOTA = ([\d\.-]+);.*?IDF1 = ([\d\.]+);"

    with open(log_path, 'r') as f:
        content = f.read()
        
        # 1. Parse Training Data
        for match in re.finditer(train_pattern, content):
            data['epoch'].append(int(match.group(1)))
            data['loss'].append(float(match.group(2)))
            data['detr_loss'].append(float(match.group(3)))
            data['id_loss'].append(float(match.group(4)))
            data['class_error'].append(float(match.group(5)))
            
            # Handle 'inf' or 'nan' in grad_norm
            grad = match.group(6)
            if 'inf' in grad.lower() or 'nan' in grad.lower():
                data['grad_norm'].append(None) # Skip plotting infinity
            else:
                data['grad_norm'].append(float(grad))

        # 2. Parse Eval Data (Align with epochs)
        eval_map = {}
        for match in re.finditer(eval_pattern, content):
            epoch = int(match.group(1))
            eval_map[epoch] = {
                'HOTA': float(match.group(2)),
                'MOTA': float(match.group(3)),
                'IDF1': float(match.group(4))
            }
        
        # align eval data to training epochs
        for e in data['epoch']:
            if e in eval_map:
                data['HOTA'].append(eval_map[e]['HOTA'])
                data['MOTA'].append(eval_map[e]['MOTA'])
                data['IDF1'].append(eval_map[e]['IDF1'])
            else:
                data['HOTA'].append(0.0)
                data['MOTA'].append(0.0)
                data['IDF1'].append(0.0)

    return data

def plot_dashboard(log_path):
    data = parse_log(log_path)
    epochs = data['epoch']
    
    if not epochs:
        print("❌ No completed epochs found in log yet.")
        return

    # Create a 2x2 Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Dashboard: {os.path.basename(os.path.dirname(log_path))}', fontsize=16)

    # Plot 1: Main Losses (Total vs DETR vs ID)
    axs[0, 0].plot(epochs, data['loss'], 'r-o', label='Total Loss')
    axs[0, 0].plot(epochs, data['detr_loss'], 'g--', label='DETR Loss')
    axs[0, 0].plot(epochs, data['id_loss'], 'b--', label='ID Loss')
    axs[0, 0].set_title('Loss Components')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation Metrics (HOTA, MOTA, IDF1)
    axs[0, 1].plot(epochs, data['HOTA'], 'b-s', linewidth=2, label='HOTA')
    axs[0, 1].plot(epochs, data['MOTA'], 'g--', label='MOTA')
    axs[0, 1].plot(epochs, data['IDF1'], 'm:', label='IDF1')
    axs[0, 1].set_title('Validation Performance')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Score (%)')
    axs[0, 1].set_ylim(0, 80) # Adjust if scores get higher
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Plot 3: Stability (Gradient Norm)
    # Filter out Nones for plotting
    valid_grads = [(e, g) for e, g in zip(epochs, data['grad_norm']) if g is not None]
    if valid_grads:
        axs[1, 0].plot(*zip(*valid_grads), 'k-x', label='Grad Norm')
        axs[1, 0].set_title('Training Stability (Gradient Norm)')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Norm')
        axs[1, 0].grid(True, alpha=0.3)
    else:
        axs[1, 0].text(0.5, 0.5, 'Gradients were Inf/NaN', ha='center')

    # Plot 4: Classifier Accuracy (Class Error)
    axs[1, 1].plot(epochs, data['class_error'], 'r-d', label='Class Error')
    axs[1, 1].set_title('Classification Error (Lower is Better)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Error Rate')
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = "dashboard.png"
    plt.savefig(output_file)
    print(f"✅ Dashboard saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str, help="Path to train.log")
    args = parser.parse_args()
    plot_dashboard(args.log_file)
