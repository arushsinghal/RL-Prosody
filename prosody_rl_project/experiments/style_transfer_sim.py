import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.simulate import run_simulation
from src.utils import plot_prosody

def run_style_adaptation():
    print("Running Experiment 3: Style/Context Adaptation")
    
    texts = [
        "The project is going very well.", # Statement
        "Is the project going very well?", # Question
        "The project is absolutely *amazing* and going perfectly!" # Excited / Emphasis
    ]
    
    for i, text in enumerate(texts):
        print(f"Adapting text: {text}")
        res = run_simulation(text, num_iterations=100)
        
        plot_prosody(
            res["tokens"], 
            res["baseline"], 
            res["optimized"], 
            title=f"Style Adaptation: {text}", 
            filename=f"exp3_style_{i}.png"
        )
        
if __name__ == "__main__":
    run_style_adaptation()
