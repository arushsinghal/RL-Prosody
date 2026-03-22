# 📚 Research & Technology Overview: RL-Prosody

This document provides a comprehensive breakdown of the features, technologies, and research methodologies used in the **RL-Prosody** project. This is a "living" research prototype designed to demonstrate how reinforcement learning can supersede supervised learning in subjective Text-to-Speech (TTS) tasks.

---

## 🏗️ 1. Project Architecture & Features

### Core Capabilities
The system allows for the **fine-grained manipulation** of prosodic parameters in a sequence:
*   **Pitch Contour**: The fundamental frequency ($F_0$) trajectory over time.
*   **Duration Scaling**: Word-to-word or syllable-to-syllable timing adjustments.
*   **Energy Contour**: The amplitude/loudness trajectory.
*   **Pause Insertion**: Strategic silences to enhance grammatical clarity or emotional impact.

### State-Space Representation
The agent observes a **state vector** derived from linguistic context:
*   `Token Sequence`: Phoneme-level or word-level identifiers.
*   `Punctuation Markers`: Commas, periods, and question marks that influence boundary prosody.
*   `Linguistic Tags`: Explicit markers for Emphasis, Context (Question, Statement, Excited).
*   `Positional Encoding`: Relative position in the sentence (vital for phrase-final lengthening).

---

## 🧪 2. Reinforcement Learning Framework

### The Reward Function ($R$)
We avoid the "average-reconstruction trap" by defining an explicit **multi-objective proxy reward**:

1.  **Naturalness ($R_{nat}$)**: Uses local smoothness (low first-derivative magnitude) balanced against global variance (standard deviation targets) to ensure speech is melodic but not jittery.
    *   *Equation Logic*: $\text{Penalty} \propto | \text{smoothness} - \text{ideal\_variation} |$
2.  **Intelligibility ($R_{int}$)**: Penalizes deviations from conservative duration bounds. If words are too short or too long, comprehension drops sharply.
    *   *Equation Logic*: $R_{int} = -\sum (d_i - 1.0)^2$
3.  **Context Alignment ($R_{ctx}$)**: Rewards specific patterns based on syntactic labels (e.g., rising pitch at the end of questions).
4.  **Monotonicity Penalty ($R_{mono}$)**: A stiff penalty for near-zero variance. This prevents the agent from collapsing into "safe" but robotic flat contours.

### The Optimization Policy
We implement a **Black-Box Optimization** strategy using **Hill Climbing with Simulated Annealing**:
*   **Exploration**: The agent injects Gaussian noise into the action trajectory.
*   **Update**: If a noisy mutation increases the scalar Reward $R$, it becomes the new "best" policy.
*   **Annealing**: The "temperature" (noise scale and learning rate) decays over time, allowing for wide exploration early and precise refinement late in the loop.

---

## 🛠️ 3. Technology Stack

### Backend: Python Ecosystem
*   **NumPy**: Used for all mathematical modeling, vector operations, and reward calculations. High performance without the overhead of heavy Deep Learning frameworks.
*   **Matplotlib**: Generates the publication-quality plots used to track reward progression and Pareto frontiers.

### Frontend: Interactive Research Dashboard
*   **HTML5/CSS3**: A minimalist, dark academic theme designed for readability and focus.
*   **Vanilla JavaScript**: Implements the "Live RL Update" logic on the client-side. This allows the user to see the optimizer oscillating and converging in real-time.
*   **Chart.js**: chosen for high-performance canvas rendering of the pitch and duration trajectories.

### Deployment & Tooling
*   **Git Subtree**: Orchestrates the automated deployment of the subfolder UI to **GitHub Pages**.
*   **Venv**: Ensures reproducible environments for the experiment scripts.

---

## 📊 4. Experimental Research Findings

### 4.1 Reward Ablation
**Finding**: Removing the *Monotonicity Penalty* caused the agent to produce perfectly smooth but completely robotic speech. Removing the *Intelligibility* constraint caused the agent to "hallucinate" high rewards by stretching pitch curves while losing linguistic timing bounds.

### 4.2 The Pareto Tradeoff Frontier
**Analysis**: By sweeping across $\lambda_{naturalness}$ vs $\lambda_{intelligibility}$, we mapped the **optimal frontier**. It proved that you cannot maximize naturalness and intelligibility simultaneously; the "sweet spot" for real-world deployment was found where the marginal gain in smoothness began to exponentially harm duration alignment.

### 4.3 Style Adaptation
**Case Study**: The policy successfully learned to insert a 200ms pause and 15Hz pitch spike before a word tagged with `EMPHASIS`, purely via reward signal, with no supervised "ground truth" recorded speech attached.

---

## 🎤 5. Interview Defense: "Why Simulated RL?"

**Interviewer**: *"Why build a simulator with proxy rewards instead of training a real E2E Speech Transformer?"*

**Defense**: 
1.  **Isolating Signals**: "End-to-end models suffer from the 'black-box' problem. By building a simulated environment with explicit proxy rewards, I can cleanly isolate the behavior of individual reward components (like Naturalness vs Intelligibility) in a way that is impossible when training a 500M parameter model on a GPU cluster."
2.  **Principled Systems Thinking**: "This project demonstrates that I understand the **architecture** of the reinforcement learning loop—how to design state spaces, formulate reward functions that avoid 'cheating', and optimize for inference latency. These are the core skills of an RL researcher, regardless of whether the acoustic generator is a simplified simulator or a heavy diffusion model."
3.  **Rapid Prototyping**: "In a real R&D lab, we always build these toy-scale simulators first to find the optimal reward composition before we commit $50,000 of compute budget to large-scale training."

---

## 🚀 6. Project Mapping (File Guide)

*   `src/prosody_env.py`: The MDP environment defining how actions affect acoustic state.
*   `src/reward.py`: The heart of the research logic (mathematical proxy functions).
*   `src/policy.py`: The implementation of the Hill-Climbing/Simulated Annealing optimizer.
*   `experiments/run_all.py`: Orchestrates all 4 research experiments.
*   `demo.html`: The interactive research system dashboard.
*   `index.html`: The problem statement and landing page.
