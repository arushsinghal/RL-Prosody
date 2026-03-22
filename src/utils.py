import numpy as np
import matplotlib.pyplot as plt
import os

# Create figures directory if it doesn't exist
os.makedirs("../figures", exist_ok=True)

def mock_tokenize(text):
    """
    Simulates a text-to-phoneme/token converter.
    Returns a list of dicts with word and token-level details.
    """
    words = text.split()
    tokens = []
    
    # Determine context based on punctuation
    context = "statement"
    if text.endswith("?"): context = "question"
    elif text.endswith("!"): context = "excited"
        
    for i, word in enumerate(words):
        # We simulate that each word maps to a few phonemes.
        # For simplicity in this demo, our 'tokens' are just words/syllables.
        token_length = max(1, len(word) // 2) 
        
        is_emphasis = word.isupper() or "*" in word
        clean_word = word.replace("*", "").replace(".", "").replace("?", "").replace("!", "").replace(",", "")
        
        tokens.append({
            "text": clean_word,
            "length": token_length,
            "is_emphasis": is_emphasis,
            "context": context,
            "is_last_word": i == len(words) - 1,
            "has_comma_after": "," in word
        })
    return tokens

def plot_prosody(tokens, prosody_baseline, prosody_rl, title="Prosody Comparison", filename=None):
    """
    Plots the pitch contour and duration map comparing baseline and RL optimization.
    prosody format: array of shape (N_tokens, 4) -> [pitch, duration, energy, pause]
    """
    labels = [t["text"] for t in tokens]
    x = np.arange(len(labels))
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # 1. Pitch
    axs[0].plot(x, prosody_baseline[:, 0], marker='o', linestyle='--', label='Baseline', color='gray')
    axs[0].plot(x, prosody_rl[:, 0], marker='o', linestyle='-', label='RL Optimized', color='blue')
    axs[0].set_ylabel("Pitch (Hz offset)")
    axs[0].set_title(title)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # 2. Duration
    width = 0.35
    axs[1].bar(x - width/2, prosody_baseline[:, 1], width, label='Baseline', color='lightgray')
    axs[1].bar(x + width/2, prosody_rl[:, 1], width, label='RL Optimized', color='orange')
    axs[1].set_ylabel("Duration Scaling")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # 3. Energy & Pauses
    axs[2].plot(x, prosody_baseline[:, 2], marker='s', linestyle='--', label='Energy (Base)', color='lightgray')
    axs[2].plot(x, prosody_rl[:, 2], marker='s', linestyle='-', label='Energy (RL)', color='red')
    
    # Plot pauses as vertical bars
    for i in range(len(x)):
        if prosody_rl[i, 3] > 0.1: # Pause > 100ms
            axs[2].axvline(i + 0.5, color='purple', alpha=0.5, linestyle=':', linewidth=prosody_rl[i, 3]*5)
            if i == 0: axs[2].axvline(i + 0.5, color='purple', alpha=0.5, linestyle=':', linewidth=1, label="Pause (RL)")

    axs[2].set_ylabel("Energy & Pauses")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, rotation=45, ha='right')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if filename:
        save_path = os.path.join(os.path.dirname(__file__), "..", "figures", filename)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    plt.close()
