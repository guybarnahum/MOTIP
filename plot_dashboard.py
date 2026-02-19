import re
import matplotlib.pyplot as plt
import argparse
import os

def parse_log(log_path):
    data = {
        'epoch': [],
        'loss': [], 'detr_loss': [], 'id_loss': [],
        'class_error': [], 'grad_norm': [],
        'HOTA': [], 'MOTA': [], 'IDF1': [],
        'DetPr': [], 'DetRe': [], 'AssA': [], 'DetA': [] 
    }
    
    # Robust Training Pattern: Handles optional spaces and interleaving text
    train_pattern = (
        r"\[Finish epoch:\s+(\d+)\].*?"
        r"loss\s+=\s+([\-\d\.]+);.*?"
        r"detr_loss\s+=\s+([\-\d\.]+);.*?"
        r"id_loss\s+=\s+([\-\d\.]+);.*?"
        r"class_error\s+=\s+([\-\d\.]+);.*?"
        r"detr_grad_norm\s+=\s+([naninf\-\d\.]+);"
    )
    
    # Robust Eval Pattern: Uses lookaheads to find metrics regardless of order or presence
    eval_epoch_pattern = r"\[Eval epoch:\s+(\d+)\]"
    metrics_to_find = ['MOTA', 'IDF1', 'HOTA', 'DetA', 'AssA', 'DetPr', 'DetRe']

    print(f"Parsing log file: {log_path}...")
    
    if not os.path.exists(log_path):
        print(f"❌ File not found: {log_path}")
        return data

    with open(log_path, 'r') as f:
        content = f.read()
        
        # 1. Parse Training Data
        for match in re.finditer(train_pattern, content):
            data['epoch'].append(int(match.group(1)))
            data['loss'].append(float(match.group(2)))
            data['detr_loss'].append(float(match.group(3)))
            data['id_loss'].append(float(match.group(4)))
            data['class_error'].append(float(match.group(5)))
            
            grad = match.group(6).lower()
            data['grad_norm'].append(None if ('inf' in grad or 'nan' in grad) else float(grad))

        # 2. Parse Eval Data (Line by Line for order independence)
        eval_map = {}
        for line in content.split('\n'):
            epoch_match = re.search(eval_epoch_pattern, line)
            if epoch_match:
                e = int(epoch_match.group(1))
                if e not in eval_map: eval_map[e] = {m: None for m in metrics_to_find}
                
                for m in metrics_to_find:
                    # Find "METRIC = VALUE;"
                    m_match = re.search(rf"{m}\s+=\s+([\-\d\.]+)", line)
                    if m_match:
                        eval_map[e][m] = float(m_match.group(1))
        
        # Align eval data to training epochs
        for e in data['epoch']:
            for key in metrics_to_find:
                val = eval_map.get(e, {}).get(key)
                data[key].append(val)

    print(f"Found {len(data['epoch'])} completed training epochs.")
    return data

def plot_dashboard(log_path):
    data = parse_log(log_path)
    epochs = data['epoch']
    
    if not epochs:
        print("❌ No completed epochs found in log! (Check regex or log format)")
        return

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    # Use dirname to get the experiment name
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(log_path)))
    fig.suptitle(f'MOTIP Multi-Class Dashboard: {exp_name}\n(Epochs 0-{max(epochs)})', fontsize=16)

    # Plot 1: Main Losses
    axs[0, 0].plot(epochs, data['loss'], 'r-o', label='Total')
    axs[0, 0].plot(epochs, data['detr_loss'], 'g--', label='DETR')
    axs[0, 0].plot(epochs, data['id_loss'], 'b:', label='ID')
    axs[0, 0].set_title('Loss Components')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Plot 2: HOTA / MOTA / IDF1
    valid_idxs = [i for i, x in enumerate(data['MOTA']) if x is not None]
    if valid_idxs:
        v_epochs = [epochs[i] for i in valid_idxs]
        axs[0, 1].plot(v_epochs, [data['IDF1'][i] for i in valid_idxs], 'm-s', label='IDF1')
        axs[0, 1].plot(v_epochs, [data['HOTA'][i] for i in valid_idxs], 'b-o', label='HOTA')
        axs[0, 1].set_ylabel('Score (%)')
        
        ax2 = axs[0, 1].twinx()
        ax2.plot(v_epochs, [data['MOTA'][i] for i in valid_idxs], 'g--', alpha=0.6, label='MOTA')
        ax2.set_ylabel('MOTA (Green)')
        # Auto-scale MOTA if it's deeply negative
        min_mota = min([data['MOTA'][i] for i in valid_idxs])
        ax2.set_ylim(min(min_mota, -10), 20)
        
        h1, l1 = axs[0, 1].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        axs[0, 1].legend(h1+h2, l1+l2, loc='lower left')

    # Plot 3: Precision/Recall (If available)
    pr_idxs = [i for i, x in enumerate(data['DetPr']) if x is not None]
    if pr_idxs:
        axs[0, 2].plot([epochs[i] for i in pr_idxs], [data['DetPr'][i] for i in pr_idxs], 'r-', label='Prec')
        axs[0, 2].plot([epochs[i] for i in pr_idxs], [data['DetRe'][i] for i in pr_idxs], 'b-', label='Rec')
        axs[0, 2].legend()
    else:
        axs[0, 2].text(0.5, 0.5, 'LiteEval (MOTA/IDF1 only)\nDetailed PR hidden', ha='center', va='center')
    axs[0, 2].set_title('Detection Quality')
    axs[0, 2].grid(True, alpha=0.3)

    # Plot 4: Stability
    valid_grads = [(e, g) for e, g in zip(epochs, data['grad_norm']) if g is not None]
    if valid_grads:
        axs[1, 0].plot(*zip(*valid_grads), 'k-', alpha=0.7)
        axs[1, 0].set_title('Grad Norm (Stability)')
    
    # Plot 5: Association vs Detection
    ass_idxs = [i for i, x in enumerate(data['AssA']) if x is not None]
    if ass_idxs:
        axs[1, 1].plot([epochs[i] for i in ass_idxs], [data['AssA'][i] for i in ass_idxs], 'purple', label='AssA')
        axs[1, 1].plot([epochs[i] for i in ass_idxs], [data['DetA'][i] for i in ass_idxs], 'orange', label='DetA')
        axs[1, 1].legend()
    else:
        axs[1, 1].text(0.5, 0.5, 'AssA/DetA not in log', ha='center', va='center')
    axs[1, 1].set_title('Tracking (Brain) vs Detection (Eyes)')

    # Plot 6: Stats Box
    axs[1, 2].axis('off')
    if epochs:
        latest_loss = data['loss'][-1]
        summary = f"LATEST STATS (Epoch {max(epochs)}):\n" \
                  f"---------------------------\n" \
                  f"Total Loss: {latest_loss:.4f}\n" \
                  f"DETR Loss: {data['detr_loss'][-1]:.4f}\n" \
                  f"ID Loss: {data['id_loss'][-1]:.4f}\n"
        if valid_idxs:
            summary += f"MOTA: {data['MOTA'][-1]:.4f}\n" \
                       f"IDF1: {data['IDF1'][-1]:.4f}"
        axs[1, 2].text(0.1, 0.5, summary, fontsize=12, family='monospace', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = os.path.join(os.path.dirname(log_path), "train_dashboard.png")
    plt.savefig(output_file)
    plt.close() # Free memory
    print(f"✅ Dashboard updated: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str)
    args = parser.parse_args()
    plot_dashboard(args.log_file)