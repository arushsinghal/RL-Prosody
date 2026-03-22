from src.utils import mock_tokenize
from src.baseline import generate_baseline_prosody
from src.prosody_env import ProsodyEnv
from src.reward import ProsodyReward
from src.policy import HillClimbingOptimizer

def run_simulation(text, reward_weights=None, num_iterations=50):
    """
    End-to-end simulation runner for a given text.
    """
    # 1. State preparation
    tokens = mock_tokenize(text)
    baseline = generate_baseline_prosody(tokens)
    
    # 2. Reward formulation
    reward_fn = ProsodyReward(weights=reward_weights)
    
    # 3. Environment
    env = ProsodyEnv(tokens, baseline, reward_fn)
    
    # 4. Policy Optimization
    optimizer = HillClimbingOptimizer(env, num_iterations=num_iterations, population_size=30, lr=2.0)
    optimized_prosody, reward_history, final_info = optimizer.optimize()
    
    return {
        "tokens": tokens,
        "baseline": baseline,
        "optimized": optimized_prosody,
        "reward_history": reward_history,
        "final_info": final_info
    }

if __name__ == "__main__":
    text = "Hello, this is a test of the reinforcement learned prosody system!"
    res = run_simulation(text)
    print("Baseline:")
    print(res["baseline"][:3])
    print("Optimized:")
    print(res["optimized"][:3])
    print("Final Reward:", res["reward_history"][-1])
