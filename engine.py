import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import diags

TOP_K = 5

# -- 1. Load Engine --
def load_engine():
    with open("./pickles/U_matrix.pkl", 'rb') as f:
        U = pickle.load(f)
    with open("./pickles/s_matrix.pkl", 'rb') as f:
        s = pickle.load(f)
    with open("./pickles/Vt_matrix.pkl", 'rb') as f:
        Vt = pickle.load(f)
    with open("./pickles/vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)
    with open("./pickles/doc_ids.pkl", 'rb') as f:
        doc_ids = pickle.load(f)
    
    return (U, s, Vt, vectorizer, doc_ids)

def search(query):
    U, s, Vt, vectorizer, doc_ids = load_engine()

    # turn the query into vector rep. using the vectorizer that was trained on the input articles
    q_vec = vectorizer.transform([query])

    # compute q_concept by projecting it on Vt.T, the term-concept matrix
    q_concept = q_vec @ Vt.T

    # obtain the document-concept search space
    docs_vec = U @ diags(s) 

    # search
    scores = cosine_similarity(q_concept, docs_vec).flatten()

    # rank the results by cosine similarity
    top_indices = scores.argsort()[::-1][:TOP_K]
        
    # 5. Display
    print(f"\nTop {TOP_K} Results:")
    for rank, idx in enumerate(top_indices):
        print(f"{rank+1}. [Score: {scores[idx]:.4f}] arXiv link: https://arxiv.org/pdf/{doc_ids[idx]}")

search("Efficient Two-Stage Group Testing Algorithms for DNA Screening")
