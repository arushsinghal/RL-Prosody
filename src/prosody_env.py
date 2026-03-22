import numpy as np

class ProsodyEnv:
    """
    A simulated RL environment for prosody manipulation.
    Given an utterance, the agent applies continuous actions space to modify the baseline prosody.
    """
    def __init__(self, tokens, baseline_prosody, reward_fn):
        self.tokens = tokens
        self.baseline_prosody = baseline_prosody.copy()
        self.current_prosody = baseline_prosody.copy()
        self.N = len(tokens)
        
        # Action space: [change_pitch, change_dur, change_energy, set_pause] for each token.
        # For a true step-by-step env, we could do it per token.
        # For modern sequence prosody models, we often generate the whole contour and refine it.
        # We will expose a `step` function that takes an action vector for the ENTIRE sequence
        # to simulate a diffusion-like or iterative refinement policy.
        self.reward_fn = reward_fn
        self.step_count = 0
        
    def reset(self):
        self.current_prosody = self.baseline_prosody.copy()
        self.step_count = 0
        return self._get_state()
        
    def _get_state(self):
        # State could incorporate text features and current prosody
        return {
            "tokens": self.tokens,
            "baseline": self.baseline_prosody,
            "current": self.current_prosody,
            "step": self.step_count
        }
        
    def step(self, action):
        """
        Action is expected to be shape (N, 4): [delta_pitch, delta_dur, delta_energy, delta_pause]
        """
        # Apply bounds/clipping to ensure realism
        self.current_prosody += action
        
        # Clip pitch to sensible values (-50, 50)
        self.current_prosody[:, 0] = np.clip(self.current_prosody[:, 0], -50, 60)
        # Duration scale > 0.5, < 2.5
        self.current_prosody[:, 1] = np.clip(self.current_prosody[:, 1], 0.5, 2.5)
        # Energy scale > 0.3, < 2.0
        self.current_prosody[:, 2] = np.clip(self.current_prosody[:, 2], 0.3, 2.0)
        # Pauses >= 0.0, <= 1.5 seconds
        self.current_prosody[:, 3] = np.clip(self.current_prosody[:, 3], 0.0, 1.5)
        
        self.step_count += 1
        
        reward, info = self.reward_fn(self._get_state())
        done = self.step_count >= 10 # Max refinement steps
        
        return self._get_state(), reward, done, info
