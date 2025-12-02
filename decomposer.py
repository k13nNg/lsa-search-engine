import pickle
import scipy.sparse.linalg 
import matplotlib.pyplot as plt

K = 100

# load the document_term_matrix
with open("./pickles/document_term_matrix.pkl", "rb") as f:
    document_term_matrix = pickle.load(f)

print(f"Matrix type: {type(document_term_matrix)}") 

def run_svd(document_term_matrix, k):
    # Since the matrix is sparse, only compute the first 100 singular values instead of all of them to save computing resource
    U, s, Vt = scipy.sparse.linalg.svds(document_term_matrix, k = k)

    # Note: svds returns them in ascending order (smallest first),
    # so we usually flip them for convenience.
    U = U[:, ::-1]
    s = s[::-1]
    Vt = Vt[::-1, :]

    return (U, s, Vt)

def generate_combined_scree(k_list):
    """
    Plots a grid of scree plots for different k values.
    
    Parameters:
    full_sigma: The complete array of singular values from SVD.
    k_list: A list of integers representing the k cut-offs (e.g., [10, 50, 100, 200, 500])
    """
    # 1. Setup the grid (2 rows, 3 columns)
    # figsize is (width, height)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    
    # Flatten the 2D array of axes into a 1D array for easy looping
    axes_flat = axes.flatten()

    # 2. Loop through your k values and plot on specific axes
    for i, k in enumerate(k_list):
        ax = axes_flat[i]
        
        # Slice the sigma array to get the top k values
        U, s, Vt = run_svd(document_term_matrix, k)

        x_axis = range(1, k + 1)
        
        # Plotting
        ax.plot(x_axis, s, 'bo-', linewidth=1.5, markersize=3)
        
        # Specific Aesthetics for this subplot
        ax.set_title(f'k = {k}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Singular Value')
        ax.set_xlabel('Latent Component')
        ax.grid(True, linestyle='--', alpha=0.6)

    # 3. Handle the empty spot (since 2x3 = 6 slots, but you only have 5 plots)
    # Hide the 6th subplot (index 5)
    axes_flat[5].axis('off')

    # 4. Global Aesthetics
    plt.suptitle('Scree Plots: Singular Values of arXiv Corpus (Varying k)', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    # 5. Save
    plt.savefig("combined_scree_plots.png", dpi=300)
    plt.show()


U, s, Vt = run_svd(document_term_matrix, K)

# save the components
with open("./pickles/U_matrix.pkl", "wb") as f:
    pickle.dump(U, f)

with open("./pickles/s_matrix.pkl", "wb") as f:
    pickle.dump(s, f)

with open("./pickles/Vt_matrix.pkl", "wb") as f:
    pickle.dump(Vt, f)