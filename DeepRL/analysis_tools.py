"""
Analysis and Monitoring Tools for DRL Maze Navigation
Provides utilities for monitoring training progress and analyzing results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob
import os
from datetime import datetime
import argparse


def load_training_stats(stats_file):
    """Load training statistics from JSON file"""
    with open(stats_file, 'r') as f:
        return json.load(f)


def plot_training_progress(stats_file, output_file=None, window=100):
    """Plot comprehensive training progress"""
    stats = load_training_stats(stats_file)
    
    rewards = np.array(stats['episode_rewards'])
    lengths = np.array(stats['episode_lengths'])
    episodes = np.arange(len(rewards))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Raw rewards
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', linewidth=0.5)
    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax1.plot(episodes[window-1:], smoothed_rewards, color='darkblue', linewidth=2, label=f'{window}-episode MA')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode lengths
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes, lengths, alpha=0.3, color='green', linewidth=0.5)
    smoothed_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
    ax2.plot(episodes[window-1:], smoothed_lengths, color='darkgreen', linewidth=2, label=f'{window}-episode MA')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Success rate over time
    ax3 = fig.add_subplot(gs[1, 0])
    success_threshold = 5.0  # Reward threshold for success
    successes = (rewards > success_threshold).astype(int)
    success_rate = np.convolve(successes, np.ones(window)/window, mode='valid') * 100
    ax3.plot(episodes[window-1:], success_rate, color='red', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title(f'Success Rate (Rolling {window} episodes)')
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% target')
    ax3.legend()
    
    # 4. Reward distribution
    ax4 = fig.add_subplot(gs[1, 1])
    # Split into early and late training
    mid_point = len(rewards) // 2
    ax4.hist(rewards[:mid_point], bins=50, alpha=0.5, label='First half', color='orange')
    ax4.hist(rewards[mid_point:], bins=50, alpha=0.5, label='Second half', color='purple')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Cumulative statistics
    ax5 = fig.add_subplot(gs[2, 0])
    cumulative_success = np.cumsum(successes) / np.arange(1, len(successes) + 1) * 100
    ax5.plot(episodes, cumulative_success, color='teal', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Cumulative Success Rate (%)')
    ax5.set_title('Cumulative Success Rate')
    ax5.set_ylim([0, 105])
    ax5.grid(True, alpha=0.3)
    
    # 6. Recent performance stats
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # Calculate stats
    recent_window = min(1000, len(rewards))
    recent_rewards = rewards[-recent_window:]
    recent_lengths = lengths[-recent_window:]
    recent_successes = successes[-recent_window:]
    
    stats_text = f"""
    Training Statistics
    {'='*40}
    
    Total Episodes: {len(rewards):,}
    Best Reward: {stats['best_reward']:.2f}
    
    Recent Performance (Last {recent_window} episodes):
    {'─'*40}
    Avg Reward: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}
    Avg Steps: {np.mean(recent_lengths):.1f} ± {np.std(recent_lengths):.1f}
    Success Rate: {np.mean(recent_successes)*100:.1f}%
    
    Overall Performance:
    {'─'*40}
    Avg Reward: {np.mean(rewards):.2f}
    Avg Steps: {np.mean(lengths):.1f}
    Overall Success: {np.mean(successes)*100:.1f}%
    
    Improvement:
    {'─'*40}
    First 1k avg: {np.mean(rewards[:1000]):.2f}
    Last 1k avg: {np.mean(recent_rewards):.2f}
    Delta: {np.mean(recent_rewards) - np.mean(rewards[:1000]):.2f}
    """
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
            fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.suptitle(f'Training Progress Analysis - {os.path.basename(stats_file)}', 
                fontsize=14, fontweight='bold')
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Training progress plot saved: {output_file}")
    else:
        plt.show()
    
    return fig


def compare_algorithms(stats_files, output_file='algorithm_comparison.png'):
    """Compare multiple algorithm training runs"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    window = 100
    
    for idx, stats_file in enumerate(stats_files):
        stats = load_training_stats(stats_file)
        algorithm = os.path.basename(stats_file).replace('training_stats_', '').replace('.json', '')
        color = colors[idx % len(colors)]
        
        rewards = np.array(stats['episode_rewards'])
        lengths = np.array(stats['episode_lengths'])
        episodes = np.arange(len(rewards))
        
        # Smoothed rewards
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(episodes[window-1:], smoothed_rewards, 
                       color=color, linewidth=2, label=algorithm, alpha=0.8)
        
        # Smoothed lengths
        smoothed_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(episodes[window-1:], smoothed_lengths, 
                       color=color, linewidth=2, label=algorithm, alpha=0.8)
        
        # Success rate
        successes = (rewards > 5.0).astype(int)
        success_rate = np.convolve(successes, np.ones(window)/window, mode='valid') * 100
        axes[1, 0].plot(episodes[window-1:], success_rate, 
                       color=color, linewidth=2, label=algorithm, alpha=0.8)
        
        # Final performance comparison (bar chart)
        recent_window = min(1000, len(rewards))
        final_success = np.mean(successes[-recent_window:]) * 100
        axes[1, 1].bar(idx, final_success, color=color, alpha=0.7, label=algorithm)
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Smoothed Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Smoothed Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_title('Success Rate Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 105])
    
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].set_title('Final Success Rate (Last 1000 episodes)')
    axes[1, 1].set_xticks(range(len(stats_files)))
    axes[1, 1].set_xticklabels([os.path.basename(f).replace('training_stats_', '').replace('.json', '') 
                                for f in stats_files], rotation=45)
    axes[1, 1].set_ylim([0, 105])
    axes[1, 1].axhline(y=80, color='green', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Algorithm Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Algorithm comparison saved: {output_file}")
    
    return fig


def analyze_evaluation_results(eval_dir):
    """Analyze evaluation results"""
    summary_file = os.path.join(eval_dir, 'evaluation_summary.json')
    
    if not os.path.exists(summary_file):
        print(f"No evaluation summary found in {eval_dir}")
        return
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    episodes = summary['episodes']
    
    # Extract data
    maze_sizes = [ep['maze_size'] for ep in episodes]
    steps = [ep['steps'] for ep in episodes]
    rewards = [ep['reward'] for ep in episodes]
    successes = [ep['success'] for ep in episodes]
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Success rate by maze size
    unique_sizes = sorted(set(maze_sizes))
    success_by_size = []
    for size in unique_sizes:
        size_successes = [s for m, s in zip(maze_sizes, successes) if m == size]
        success_by_size.append(np.mean(size_successes) * 100)
    
    axes[0, 0].bar(unique_sizes, success_by_size, color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('Maze Size')
    axes[0, 0].set_ylabel('Success Rate (%)')
    axes[0, 0].set_title('Success Rate by Maze Size')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 105])
    
    # 2. Steps vs Maze Size
    axes[0, 1].scatter(maze_sizes, steps, alpha=0.6, color='green')
    axes[0, 1].set_xlabel('Maze Size')
    axes[0, 1].set_ylabel('Steps Taken')
    axes[0, 1].set_title('Steps vs Maze Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(maze_sizes, steps, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(unique_sizes, p(unique_sizes), "r--", alpha=0.8, linewidth=2)
    
    # 3. Reward distribution
    success_rewards = [r for r, s in zip(rewards, successes) if s]
    fail_rewards = [r for r, s in zip(rewards, successes) if not s]
    
    axes[1, 0].hist(success_rewards, bins=20, alpha=0.6, label='Success', color='green')
    axes[1, 0].hist(fail_rewards, bins=20, alpha=0.6, label='Failure', color='red')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary statistics
    axes[1, 1].axis('off')
    
    stats_text = f"""
    Evaluation Summary
    {'='*40}
    
    Model: {summary['algorithm'].upper()}
    Total Episodes: {summary['num_episodes']}
    
    Overall Performance:
    {'─'*40}
    Success Rate: {summary['success_rate']*100:.1f}%
    Avg Steps: {summary['avg_steps']:.1f}
    Avg Reward: {summary['avg_reward']:.2f}
    
    By Maze Size:
    {'─'*40}
    """
    
    for size in unique_sizes:
        size_episodes = [ep for ep in episodes if ep['maze_size'] == size]
        size_success = np.mean([ep['success'] for ep in size_episodes]) * 100
        size_steps = np.mean([ep['steps'] for ep in size_episodes])
        stats_text += f"  {size}x{size}: {size_success:.0f}% success, {size_steps:.0f} steps\n"
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                   fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.suptitle(f'Evaluation Analysis - {summary["algorithm"].upper()}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(eval_dir, 'evaluation_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Evaluation analysis saved: {output_file}")
    
    return fig


def monitor_training_live(stats_file, refresh_interval=60):
    """Monitor training progress in real-time"""
    import time
    
    print(f"Monitoring training: {stats_file}")
    print(f"Refresh interval: {refresh_interval}s")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            if os.path.exists(stats_file):
                stats = load_training_stats(stats_file)
                rewards = stats['episode_rewards']
                
                if len(rewards) > 0:
                    recent_window = min(100, len(rewards))
                    recent_rewards = rewards[-recent_window:]
                    
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Episode: {len(rewards):,} | "
                          f"Recent avg reward: {np.mean(recent_rewards):.2f} | "
                          f"Best: {stats['best_reward']:.2f}", end='')
            else:
                print(f"\rWaiting for stats file...", end='')
            
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def generate_report(model_dir, output_file='training_report.html'):
    """Generate comprehensive HTML report"""
    # Find all stats files
    stats_files = glob.glob(os.path.join(model_dir, 'training_stats_*.json'))
    
    if not stats_files:
        print(f"No training stats found in {model_dir}")
        return
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DRL Maze Navigation - Training Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; }
            h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
            h2 { color: #666; margin-top: 30px; }
            .stat-box { display: inline-block; margin: 10px; padding: 15px; background-color: #f0f0f0; 
                       border-radius: 5px; min-width: 150px; }
            .stat-label { font-size: 12px; color: #666; }
            .stat-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #4CAF50; color: white; }
            .success { color: green; font-weight: bold; }
            .failure { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Deep RL Maze Navigation - Training Report</h1>
            <p>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    """
    
    for stats_file in stats_files:
        stats = load_training_stats(stats_file)
        algorithm = os.path.basename(stats_file).replace('training_stats_', '').replace('.json', '')
        
        rewards = np.array(stats['episode_rewards'])
        lengths = np.array(stats['episode_lengths'])
        successes = (rewards > 5.0).astype(int)
        
        recent_window = min(1000, len(rewards))
        
        html_content += f"""
            <h2>{algorithm.upper()} Training Results</h2>
            
            <div class="stat-box">
                <div class="stat-label">Total Episodes</div>
                <div class="stat-value">{len(rewards):,}</div>
            </div>
            
            <div class="stat-box">
                <div class="stat-label">Best Reward</div>
                <div class="stat-value">{stats['best_reward']:.2f}</div>
            </div>
            
            <div class="stat-box">
                <div class="stat-label">Recent Avg Reward</div>
                <div class="stat-value">{np.mean(rewards[-recent_window:]):.2f}</div>
            </div>
            
            <div class="stat-box">
                <div class="stat-label">Success Rate</div>
                <div class="stat-value">{np.mean(successes[-recent_window:])*100:.1f}%</div>
            </div>
            
            <div class="stat-box">
                <div class="stat-label">Avg Steps</div>
                <div class="stat-value">{np.mean(lengths[-recent_window:]):.0f}</div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analysis tools for DRL maze navigation')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['plot', 'compare', 'evaluate', 'monitor', 'report'],
                       help='Analysis mode')
    parser.add_argument('--stats_file', type=str,
                       help='Path to training stats JSON file')
    parser.add_argument('--stats_files', type=str, nargs='+',
                       help='Multiple stats files for comparison')
    parser.add_argument('--eval_dir', type=str,
                       help='Evaluation results directory')
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Model directory')
    parser.add_argument('--output', type=str,
                       help='Output file path')
    
    args = parser.parse_args()
    
    if args.mode == 'plot':
        if not args.stats_file:
            print("Error: --stats_file required for plot mode")
        else:
            plot_training_progress(args.stats_file, args.output)
    
    elif args.mode == 'compare':
        if not args.stats_files:
            print("Error: --stats_files required for compare mode")
        else:
            compare_algorithms(args.stats_files, args.output or 'comparison.png')
    
    elif args.mode == 'evaluate':
        if not args.eval_dir:
            print("Error: --eval_dir required for evaluate mode")
        else:
            analyze_evaluation_results(args.eval_dir)
    
    elif args.mode == 'monitor':
        if not args.stats_file:
            print("Error: --stats_file required for monitor mode")
        else:
            monitor_training_live(args.stats_file)
    
    elif args.mode == 'report':
        generate_report(args.model_dir, args.output or 'training_report.html')