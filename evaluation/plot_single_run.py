import matplotlib.pyplot as plt
import numpy as np
import os

def generate_run_plots(dataset_name, model_name, exec_time, pre_target, pre_retain, post_target, post_retain, target_probs=None, save_dir="evaluation/plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a 1x2 grid if probabilities are provided, otherwise just a single plot
    if target_probs is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax2 = None

    # Plot 1: Pre vs Post Unlearning Accuracy
    labels = ['Target Class', 'Retained Classes']
    pre_accs = [pre_target, pre_retain]
    post_accs = [post_target, post_retain]
    
    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width/2, pre_accs, width, label='Pre-Unlearning', color='#2c3e50')
    ax1.bar(x + width/2, post_accs, width, label='Post-Unlearning', color='#e74c3c')

    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'SSN Performance: {model_name} on {dataset_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 110)
    ax1.legend()
    
    # Plot 2: Final Probability Distribution (The Shannon Entropy Proof)
    if ax2 is not None:
        classes = np.arange(len(target_probs))
        
        # Convert probabilities to percentages for the chart
        ax2.bar(classes, target_probs * 100, color='#3498db')
        
        # Draw a red dashed line at the ideal 10% uniform distribution target
        ax2.axhline(y=10, color='r', linestyle='--', label='Ideal Uniform (10%)')
        
        ax2.set_xlabel('Class Index')
        ax2.set_ylabel('Confidence Probability (%)')
        ax2.set_title('Post-Unlearning Output Distribution (Target Image)')
        ax2.set_xticks(classes)
        ax2.set_ylim(0, max(100, max(target_probs * 100) + 10))
        ax2.legend()

    plt.suptitle(f'Execution Time: {exec_time:.2f}s', fontsize=12, y=0.98)
    plt.tight_layout()
    
    # Save the figure
    filename = f"{dataset_name}_{model_name.replace(' ', '_')}_evaluation.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    print(f"Plots saved successfully to {filepath}")
    plt.close()