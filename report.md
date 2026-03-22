# Reinforcement-Learned Adaptive Prosody for Text-to-Speech

## Objective
The objective of this project is to demonstrate how reinforcement learning (RL) can adapt and optimize prosody—specifically pitch, duration, energy, and pauses—for Text-to-Speech (TTS) systems. By employing a simulation environment and a simple evolutionary optimization policy, this framework treats prosody generation as a reward-driven trajectory optimization problem.

## Motivation
Prosody is notoriously difficult to model in traditional TTS systems. Most models rely on supervised learning (e.g., L1/L2 loss on mel-spectrograms), which often leads to average, monotonous speech because prosody is highly subjective and context-dependent. A single sentence can be spoken correctly in ten different ways. Supervised reconstruction loss penalizes variation, whereas Reinforcement Learning allows a model to optimize for explicit, composite qualities such as naturalness, intelligibility, and contextual appropriateness.

## Method
- **Environment**: A custom `ProsodyEnv` steps through tokens and receives continuous action updates modifying baseline heuristics.
- **State/Action/Reward**: 
  - *State*: Linguistic context (phoneme length, punctuation, emphasis markings).
  - *Action*: Pitch offset, duration scaling, energy scaling, pause insertion constraint.
  - *Reward*: A multi-component proxy including Naturalness (smoothness with variance), Intelligibility (target duration bounds), Context Match (e.g., questions end with a rising pitch), and Monotonicity Penalties.
- **Policy/Optimization**: A lightweight Cross-Entropy/Hill Climbing continuous optimizer simulating step-by-step policy refinement.
- **Evaluation**: Assessed using quantitative ablation metrics and Pareto tradeoff visualization.

## Experiments
1. **Reward Ablation**: Compared isolated reward components against a balanced multi-objective reward function to observe resulting parameter contours.
2. **Prosody Tradeoff Frontier**: Swept the weight between Naturalness and Intelligibility to plot the Pareto frontier (optimal tradeoffs).
3. **Style/Context Adaptation**: Evaluated the policy on identical text but different linguistic tags (Statement, Question, Emphasis) to ensure contextual divergence.
4. **Latency vs Quality**: Simulated how successive inference refinements increase prosody quality but linearly increase compute latency.

## Results
- Optimizing purely for naturalness often caused durations to drift toward extremes (creating overly fast/slow but melodically smooth speech).
- Optimizing purely for intelligibility flattened pitch and removed pauses.
- The tradeoff curve (Experiment 2) clearly shows an optimal "sweet spot" at $\lambda \approx 0.4$.

## Key Insights
1. **Reward design strongly shapes prosody behavior**: Just like real RL, the agent will exploit poor proxy rewards. A balance is necessary to avoid "intelligible but robotic" or "natural but incoherent".
2. **Naturalness/intelligibility tradeoff exists**: Too much variance harms intelligibility; too little variance harms naturalness.
3. **Context-aware prosody is essential**: The system successfully mapped rising pitch to questions and high energy to emphasized words entirely due to the contextual reward.
4. **Extra refinement can improve quality but hurts latency**: There are sharp diminishing returns after ~20-50 iterations in the simulation, indicating dynamic early-stopping could benefit real TTS systems.

## Limitations
- **Proxy Rewards**: This project relies on heuristic mathematical proxies rather than real Human-in-the-Loop Feedback (RLHF) or large pre-trained discriminator networks.
- **Simulated Environment**: Actions are applied to pre-computed duration/pitch tracks rather than directly backpropagating through an acoustic model.
- **Not Full E2E TTS**: This is an isolated optimization of the prosodic variables, distinct from the waveform vocoder or spectrogram generator.
