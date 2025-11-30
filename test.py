import numpy as np
import pickle

def interpret_concepts():
    # Load the dictionary maps
    with open("./pickles/Vt_matrix.pkl", "rb") as f:
        Vt = pickle.load(f)
    with open("./pickles/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
        
    # Get the actual words array ["algorithm", "binary", ...]
    vocab = vectorizer.get_feature_names_out()
    
    print("\n--- Latent Concept Interpretation ---")
    
    # Look at the first 5 dimensions (The "Loudest" concepts)
    for i in range(5):
        # Get the row from Vt corresponding to Concept i
        component = Vt[i]
        
        # Sort indices by the weight (absolute value usually matters most in SVD, 
        # but let's look at the strongest positive correlations first)
        top_indices = component.argsort()[::-1][:10]
        
        # Extract the words
        top_words = [vocab[idx] for idx in top_indices]
        
        print(f"\nDimension {i+1} (Variance: High):")
        print(f"  Top Words: {', '.join(top_words)}")
        
        # Optional: Print what you THINK this topic is
        # e.g., if you see "neural, network, layer", it's DL.

if __name__ == "__main__":
    interpret_concepts()