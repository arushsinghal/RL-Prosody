import numpy as np

class ProsodyReward:
    """
    Reward function that aggregates multiple proxy components.
    """
    def __init__(self, weights=None):
        # Default weights for ablation
        self.weights = {
            "naturalness": 1.0,
            "intelligibility": 1.0,
            "context_match": 1.0,
            "monotonicity_penalty": 1.0,
            "latency_penalty": 0.1
        }
        if weights:
            self.weights.update(weights)
            
    def compute(self, state):
        tokens = state["tokens"]
        baseline = state["baseline"]
        current = state["current"]
        step = state["step"]
        
        N = len(tokens)
        
        ## 1. Naturalness Proxy
        # Real human speech has smooth pitch transitions but isn't a straight line.
        # We penalize extreme high-frequency changes (jitter) but reward some variance.
        pitch = current[:, 0]
        pitch_diffs = np.diff(pitch)
        smoothness_penalty = np.sum(np.abs(pitch_diffs)) # L1 variation
        # We don't want it to be 0 (monotonic), so ideal variation is say, N * 5 Hz.
        ideal_variation = N * 5.0
        naturalness = -0.1 * abs(smoothness_penalty - ideal_variation)
        # Also penalize moving TOO far from baseline to avoid insane sounds
        div_penalty = -0.05 * np.sum(np.abs(current - baseline))
        r_nat = naturalness + div_penalty
        
        ## 2. Intelligibility Proxy
        # Primarily affected by duration (talking too fast = garbled).
        durations = current[:, 1]
        # Ideal duration is around 1.0 scale.
        dur_penalty = -np.sum((durations - 1.0)**2)
        r_int = dur_penalty
        
        ## 3. Context Match
        r_ctx = 0.0
        for i, t in enumerate(tokens):
            if t["context"] == "question" and t["is_last_word"]:
                r_ctx += pitch[i] # Reward high pitch at end of question
            if t["is_emphasis"]:
                r_ctx += (current[i, 2] * 5 + pitch[i] * 0.5) # Reward high energy & pitch
            if t["context"] == "excited":
                r_ctx += (current[i, 2] * 2 + np.mean(pitch) * 0.1)
                
        ## 4. Monotonicity Penalty
        # Heavily penalize if standard deviation of pitch/energy is too low
        std_p = np.std(pitch) + 1e-6
        r_mono = -5.0 / std_p
        
        ## 5. Latency Penalty
        # Penalize for more refinement steps
        r_lat = -float(step)
        
        total_reward = (
            self.weights["naturalness"] * r_nat +
            self.weights["intelligibility"] * r_int +
            self.weights["context_match"] * r_ctx +
            self.weights["monotonicity_penalty"] * r_mono +
            self.weights["latency_penalty"] * r_lat
        )
        
        components = {
            "r_nat": r_nat,
            "r_int": r_int,
            "r_ctx": r_ctx,
            "r_mono": r_mono,
            "r_lat": r_lat,
            "smoothness": smoothness_penalty
        }
        
        return total_reward, components

    def __call__(self, state):
        return self.compute(state)
