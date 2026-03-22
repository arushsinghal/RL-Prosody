import numpy as np

def generate_baseline_prosody(tokens):
    """
    Generates heuristic-based baseline prosody for a sequence of tokens.
    Returns a numpy array of shape (N, 4).
    Features: [pitch_offset (Hz), duration_scale, energy_scale, pause_after (seconds)]
    """
    N = len(tokens)
    prosody = np.zeros((N, 4))
    
    base_pitch_offset = 0.0
    base_dur = 1.0
    base_energy = 1.0
    
    for i, t in enumerate(tokens):
        # Default
        p = base_pitch_offset
        d = t["length"] * base_dur
        e = base_energy
        pause = 0.0
        
        # Heuristic rules
        if t["context"] == "statement":
            # Declining pitch over the sentence
            p = 10.0 - (i / max(1, N-1)) * 20.0 
        elif t["context"] == "question":
            # Rising pitch at the end
            if t["is_last_word"]:
                p = 30.0
            else:
                p = 5.0
        elif t["context"] == "excited":
            # Higher energy and pitch overall
            p = 20.0 + np.random.uniform(-5, 5)
            e = 1.4
            
        if t["is_emphasis"]:
            p += 15.0
            e += 0.3
            d *= 1.2
            
        if t["has_comma_after"]:
            pause = 0.2
            p += 5.0 # slight rise before pause
            
        if t["is_last_word"] and t["context"] != "question":
            pause = 0.5
            e *= 0.8 # fade out
            d *= 1.2 # phrase final lengthening
            
        prosody[i] = [p, d, e, pause]
        
    return prosody
