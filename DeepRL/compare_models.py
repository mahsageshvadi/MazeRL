import json
import matplotlib.pyplot as plt
import numpy as np
import glob

def plot_training_comparison(model_dirs):
    """Compare training curves from multiple models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for model_dir in model_dirs:
        stats_files = glob.glob(f"{model_dir}/training_stats_*.json")
        
        for stats_file in stats_files:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            algorithm = stats_file.split('_')[-1].replace('.json', '')
            
            # Plot rewards
            window = 100
            rewards = stats['episode_rewards']
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(smoothed, label=f"{algorithm} - {model_dir}")
            
            # Plot episode lengths
            lengths = stats['episode_lengths']
            smoothed_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(smoothed_len, label=f"{algorithm} - {model_dir}")
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (smoothed)')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps (smoothed)')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150)
    print("Comparison plot saved: training_comparison.png")

if __name__ == "__main__":
    # Compare models from different runs
    model_dirs = ['./models', './models_run2']
    plot_training_comparison(model_dirs)
