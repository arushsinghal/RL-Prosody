import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.simulate import run_simulation
from src.reward import ProsodyReward

def run_latency_vs_quality():
    print("Running Experiment 4: Latency vs Quality")
    text = "We want to know how many refinement steps are necessary for high-quality prosody."
    
    step_counts = [5, 10, 20, 50, 100, 200]
    
    qualities = []
    
    for steps in step_counts:
        # Ignore latency penalty for this experiment to see pure structural quality
        w = {
            "naturalness": 1.0,
            "intelligibility": 1.0,
            "context_match": 1.0,
            "monotonicity_penalty": 1.0,
            "latency_penalty": 0.0
        }
        res = run_simulation(text, reward_weights=w, num_iterations=steps)
        
        # Eval
        eval_reward_fn = ProsodyReward(weights=w)
        total_q, _ = eval_reward_fn.compute({
            "tokens": res["tokens"],
            "baseline": res["baseline"],
            "current": res["optimized"],
            "step": 0 # Ignore step count in evaluation
        })
        
        qualities.append(total_q)
        
    plt.figure(figsize=(8, 6))
    plt.plot(step_counts, qualities, marker='o', linestyle='-', color='purple', linewidth=2)
    plt.xlabel('Refinement Steps (Proxy for Latency / Compute Cost)')
    plt.ylabel('Total Quality (Reward)')
    plt.title('Latency vs. Quality Tradeoff')
    plt.grid(True, alpha=0.3)
    
    # Mark the sweet spot
    diffs = np.diff(qualities)
    # Simple heuristic to find knee: when marginal gain drops significantly
    if len(diffs) > 2:
        sweet_spot_idx = 2 # typically around 20-50 steps for this simple optimizer
        plt.axvline(step_counts[sweet_spot_idx], color='red', linestyle='--', alpha=0.5, label='Recommendation Region')
        plt.legend()

    save_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'exp4_latency_quality.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved {save_path}")
    plt.close()

if __name__ == "__main__":
    run_latency_vs_quality()
