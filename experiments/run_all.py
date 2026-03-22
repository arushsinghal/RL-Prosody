import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.reward_ablation import run_reward_ablation
from experiments.prosody_tradeoff import run_tradeoff_experiment
from experiments.style_transfer_sim import run_style_adaptation
from experiments.latency_vs_quality import run_latency_vs_quality

def main():
    print("="*50)
    print("RUNNING PROSODY RL EXPERIMENTS")
    print("="*50)
    
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'figures'), exist_ok=True)
    
    run_reward_ablation()
    print("-" * 30)
    run_tradeoff_experiment()
    print("-" * 30)
    run_style_adaptation()
    print("-" * 30)
    run_latency_vs_quality()
    print("-" * 30)
    
    print("All experiments completed! Check the 'figures' directory.")

if __name__ == "__main__":
    main()
