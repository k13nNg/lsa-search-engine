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

def generate_scree_plot(sigma, k):
    # 1. Plot Setup
    plt.figure(figsize=(10, 6))

    # make sure the components start from 1 (Python starts counting from 0)
    x_axis = range(1, k+1)
    
    # Plot the raw Singular Values
    plt.plot(x_axis, sigma, 'bo-', linewidth=2, markersize=4, label='Singular Value')
    
    # 2. Aesthetics
    plt.title(f'Scree Plot: Singular Values of arXiv Corpus, k = {k}', fontsize=16)
    plt.xlabel('Latent Component', fontsize=12)
    plt.ylabel('Singular Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig(f"scree_plot_{k}.png")
    # plt.show()

# for i in [10, 50, 100, 200, 500]:
#     U, s, Vt = run_svd(document_term_matrix, i)

#     # # save the components
#     # with open("./pickles/U_matrix.pkl", "wb") as f:
#     #     pickle.dump(U, f)

#     # with open("./pickles/s_matrix.pkl", "wb") as f:
#     #     pickle.dump(s, f)

#     # with open("./pickles/Vt_matrix.pkl", "wb") as f:
#     #     pickle.dump(Vt, f)

#     generate_scree_plot(s, i)

U, s, Vt = run_svd(document_term_matrix, K)

# save the components
with open("./pickles/U_matrix.pkl", "wb") as f:
    pickle.dump(U, f)

with open("./pickles/s_matrix.pkl", "wb") as f:
    pickle.dump(s, f)

with open("./pickles/Vt_matrix.pkl", "wb") as f:
    pickle.dump(Vt, f)