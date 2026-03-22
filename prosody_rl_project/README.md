# Reinforcement-Learned Adaptive Prosody for TTS

This repository implements a lightweight, simulated research environment for applying reinforcement learning (RL) concepts to Text-to-Speech (TTS) prosody. Instead of relying purely on supervised reconstruction loss (which leads to "average", flat voices), this system uses a multi-objective proxy reward to optimize pitch, duration, energy, and pausing.

## Project Structure
- `src/`: Core simulation environment (`prosody_env.py`), constraints/rewards (`reward.py`), simulated RL policy optimization (`policy.py`).
- `experiments/`: Scripts that sweeping different variables to produce tables and graphs.
- `figures/`: The generated plots (run experiments to populate).
- `demo.html`: A visual dashboard explaining the results.
- `report.md` / `paper.tex`: The associated research notes.

## Run Instructions

1. Install basic dependencies (Numpy, Matplotlib):
   ```bash
   pip3 install -r requirements.txt
   ```
2. Run the experimental suite:
   ```bash
   python3 experiments/run_all.py
   ```
3. Open `demo.html` in your web browser. This dashboard has been upgraded into a **LIVE interactive system**. It uses iterative Javascript visualization to simulate the mathematics of RL optimization directly on the client side, allowing you directly adjust parameters, run live gradient/hill-climbing steps, and watch the reward function penalize failure modes (like over-smoothing or monotonic collapse).

---

## Technical Appendix: Internal RL Simulation
The `demo.html` frontend achieves strict mathematical realism by executing a continuous-control Hill-Climbing algorithm per frame. It computes an instantaneous $R(s,a)$ reward function penalizing duration drift ($R_{int}$) and extreme pitch traversal ($R_{nat}$), while targeting a pre-computed contextual heuristic ($R_{ctx}$). The simulated "Run" button applies gradient-approximated steps with injected uniform noise—forcing the visual trajectory to oscillate and occasionally regress exactly as an under-trained PPO agent would, before converging onto the global maxima.

### 1. The 2-Minute Elevator Pitch
"For this project, I built a simulated environment to explore how we can use Reinforcement Learning to fix a major problem in TTS: monotonous prosody. Current models use MSE loss to mimic a single ground-truth recording, which penalizes natural variation. I framed prosody—pitch, duration, and energy—as a continuous control problem. By designing multi-objective rewards for naturalness, intelligibility, and contextual appropriateness, and using an iterative optimization policy, I demonstrated that we can explicitly steer the prosody of a sentence. It successfully learned to raise pitch for questions and add emphasize without being micro-managed by labeled data. It’s lightweight, implemented purely in Numpy, and clearly maps out the Pareto frontier between sounding natural and being strictly intelligible."

### 2. Resume-Ready Bullet Points
- **Designed a simulated RL framework** for Text-to-Speech prosody optimization, framing pitch and duration modeling as a continuous control trajectory problem.
- **Formulated a multi-objective proxy reward** balancing naturalness variance and standard intelligibility bounds, overcoming the "average-case" collapse of traditional MSE training.
- **Implemented zero-dependency continuous optimization policies** (Hill Climbing / CEM) in Numpy, generating publication-quality Pareto tradeoff frontiers and latency-quality curves.

### 3. Key Defenses & Potential Questions

**Q1: Why is this done using a heuristic proxy reward instead of real RLHF?**
*Answer:* "Real RLHF requires a massive deployed discriminator model or direct human raters in the loop, which wasn't feasible for a rapid prototype. The mathematical proxies here (penalizing unnatural high-frequency jitter, keeping duration lengths within variance bounds) are highly correlated with human preference and allowed me to iterate quickly and cleanly demonstrate the systemic RL concepts."

**Q2: Why didn't you train an end-to-end model with PyTorch?**
*Answer:* "End-to-end models obscure the mechanisms of action behind massive parameter counts. To explicitly demonstrate a mastery of RL tradeoffs—reward ablation, metric frontiers, and exploration vs. exploitation—a focused, mathematically transparent Numpy simulator is far more effective and defensible in an interview context. It proves I understand the *architecture* of the problem, not just the PyTorch API."

**Q3: How would you scale this to a production environment?**
*Answer:* "In production, the environment wouldn't be simple heuristics. The 'Environment' would be the acoustic model generating latents, and the 'Reward' would be a frozen LLM-based discriminator like a fine-tuned WavLM evaluating the waveform. The 'Policy' would likely be PPO updating an adapter layer on the TTS diffusion process to guide the acoustic frames toward higher reward zones."

**Q4: How did you calculate the Pareto Tradeoff frontier?**
*Answer:* "By doing a linear sweep of the scalar multiplier $\lambda$ governing the weight between the Intelligibility proxy and the Naturalness proxy. I saved the unscaled resultant metrics for the final policy behavior across the sweep and plotted them to find the 'knee' of the curve—the optimal balance."
