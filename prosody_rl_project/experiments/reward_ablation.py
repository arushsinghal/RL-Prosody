import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.simulate import run_simulation
from src.reward import ProsodyReward

def run_reward_ablation():
    print("Running Experiment 1: Reward Ablation")
    text = "Most TTS systems learn prosody from supervised data, but prosody is subjective."
    
    configs = {
        "Balanced (All)": {"naturalness": 1.0, "intelligibility": 1.0, "context_match": 1.0, "monotonicity_penalty": 1.0, "latency_penalty": 0.1},
        "Only Naturalness": {"naturalness": 1.0, "intelligibility": 0.0, "context_match": 0.0, "monotonicity_penalty": 0.0, "latency_penalty": 0.0},
        "Only Intelligibility": {"naturalness": 0.0, "intelligibility": 1.0, "context_match": 0.0, "monotonicity_penalty": 0.0, "latency_penalty": 0.0},
        "Ignore Monotony": {"naturalness": 1.0, "intelligibility": 1.0, "context_match": 1.0, "monotonicity_penalty": 0.0, "latency_penalty": 0.1},
    }
    
    results = {}
    tokens = None
    baseline = None
    
    for name, w in configs.items():
        print(f"Testing config: {name}")
        res = run_simulation(text, reward_weights=w, num_iterations=100)
        
        # Evaluate final state using the balanced reward to compare apples-to-apples
        eval_reward_fn = ProsodyReward(weights=configs["Balanced (All)"])
        _, components = eval_reward_fn.compute({
            "tokens": res["tokens"],
            "baseline": res["baseline"],
            "current": res["optimized"],
            "step": 100
        })
        
        results[name] = {
            "prosody": res["optimized"],
            "metrics": components
        }
        tokens = res["tokens"]
        baseline = res["baseline"]
        
    # Plotting
    labels = list(configs.keys())
    nat_scores = [results[name]["metrics"]["r_nat"] for name in labels]
    int_scores = [results[name]["metrics"]["r_int"] for name in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, nat_scores, width, label='Naturalness Proxy', color='skyblue')
    rects2 = ax.bar(x + width/2, int_scores, width, label='Intelligibility Proxy', color='salmon')
    
    ax.set_ylabel('Reward Component Score')
    ax.set_title('Reward Ablation Impact on Final Policy Quality')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'exp1_reward_ablation.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved {save_path}")
    plt.close()

if __name__ == "__main__":
    run_reward_ablation()
