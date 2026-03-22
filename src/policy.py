import numpy as np

class HillClimbingOptimizer:
    """
    A simple continuous action optimization policy.
    We use simulated annealing / hill climbing to represent the "RL Update".
    Since our environment is a differentiable-like sequence optimization problem,
    this works well to demonstrate reward-driven behavior.
    """
    def __init__(self, env, num_iterations=50, population_size=20, lr=0.1):
        self.env = env
        self.num_iterations = num_iterations
        self.pop_size = population_size
        self.lr = lr
        
    def optimize(self):
        # The true state is just the sequence length
        N = self.env.N
        
        # We are optimizing an action sequence: (N, 4)
        best_action = np.zeros((N, 4))
        
        # Evaluate initial
        self.env.reset()
        _, best_reward, _, best_info = self.env.step(best_action)
        
        history = [best_reward]
        
        for i in range(self.num_iterations):
            # Generate population of noisy actions around the best action
            # Noise scale decreases over time (simulated annealing)
            noise_scale = max(0.01, 1.0 - (i / self.num_iterations)) * self.lr
            
            population = [best_action + np.random.normal(0, noise_scale, size=(N, 4)) for _ in range(self.pop_size)]
            
            pop_rewards = []
            pop_infos = []
            
            for act in population:
                self.env.reset()
                _, reward, _, info = self.env.step(act)
                pop_rewards.append(reward)
                pop_infos.append(info)
                
            best_idx = np.argmax(pop_rewards)
            if pop_rewards[best_idx] > best_reward:
                best_reward = pop_rewards[best_idx]
                best_action = population[best_idx]
                best_info = pop_infos[best_idx]
                
            history.append(best_reward)
            
        # Final step with best action
        self.env.reset()
        state, reward, done, info = self.env.step(best_action)
        return state["current"], history, info
