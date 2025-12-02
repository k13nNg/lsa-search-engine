import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import diags

TOP_K = 5

# -- Load Components --
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
    
def get_doc_points(n):
    # return the first n points of the document-concept matrix, along with the document ids
    return ((U @ diags(s))[:n], doc_ids[:n])

def search(query):
    # turn the query into vector rep. using the vectorizer that was trained on the input articles
    q_vec = vectorizer.transform([query])

    if q_vec.nnz == 0:
        return None
    else:
        # compute q_concept by projecting it on Vt.T, the term-concept matrix
        q_concept = q_vec @ Vt.T

        # obtain the document-concept search space
        docs_vec = U @ diags(s) 

        # search
        scores = cosine_similarity(q_concept, docs_vec).flatten()

        # rank the results by cosine similarity
        top_indices = scores.argsort()[::-1][:TOP_K]

        result = []
            
        for rank, idx in enumerate(top_indices):
            data = (rank+1, scores[idx], doc_ids[idx])
            result.append(data)

        return result, (q_concept[0,0], q_concept[0,1], q_concept[0,2])
