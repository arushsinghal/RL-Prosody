import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.simulate import run_simulation
from src.reward import ProsodyReward

def run_tradeoff_experiment():
    print("Running Experiment 2: Prosody Tradeoff Frontier")
    text = "This experiment shows the Pareto frontier between naturalness and intelligibility."
    
    lambdas = np.linspace(0.0, 1.0, 15)
    
    nat_scores = []
    int_scores = []
    
    for l in lambdas:
        # l = 0 means only naturalness, l = 1 means only intelligibility
        w = {
            "naturalness": 1.0 - l,
            "intelligibility": l,
            "context_match": 0.5,
            "monotonicity_penalty": 0.5,
            "latency_penalty": 0.0
        }
        res = run_simulation(text, reward_weights=w, num_iterations=80)
        
        # Eval using unscaled metrics
        eval_reward_fn = ProsodyReward()
        _, comps = eval_reward_fn.compute({
            "tokens": res["tokens"],
            "baseline": res["baseline"],
            "current": res["optimized"],
            "step": 80
        })
        
        nat_scores.append(comps["r_nat"])
        int_scores.append(comps["r_int"])
        
    plt.figure(figsize=(8, 6))
    plt.scatter(int_scores, nat_scores, c=lambdas, cmap='viridis', s=100, edgecolor='k')
    plt.colorbar(label='Lambda (Intelligibility Weight)')
    plt.xlabel('Intelligibility (Duration Consistency)')
    plt.ylabel('Naturalness (Smoothness & Variance)')
    plt.title('Tradeoff Frontier: Naturalness vs. Intelligibility')
    plt.grid(True, alpha=0.3)
    
    # Annotate a few points
    plt.annotate("Overly smooth\n(Unintelligible)", (int_scores[0], nat_scores[0]), textcoords="offset points", xytext=(10,-10), ha='center')
    plt.annotate("Robotic\n(Highly Intelligible)", (int_scores[-1], nat_scores[-1]), textcoords="offset points", xytext=(-10,-20), ha='center')
    
    save_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'exp2_pareto_frontier.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved {save_path}")
    plt.close()

if __name__ == "__main__":
    run_tradeoff_experiment()
