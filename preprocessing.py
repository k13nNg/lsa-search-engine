import json 
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import pickle
import random

# only get 50k documents of the filtered JSON file for ease of representation and processing
# the 50k documents will be sampled randomly to get a mix of documents from all fields
MAX_DOC = 50000
FILE_PATH = "./data/arxiv-metadata-filtered.json"

# -- 1. Random Sampling --
print(f"Randomly sampling {MAX_DOC} documents from the dataset")
all_lines = []

with open(FILE_PATH, 'r') as f:
    all_lines = f.readlines()

# randomly sample MAX_DOC 
sampled_docs = random.sample(all_lines, MAX_DOC)

# -- 2. Process JSON --
print(f"Processing {MAX_DOC} documents")
corpus = []
doc_ids = []

for line in sampled_docs:
    raw_doc = json.loads(line)
    text = f"{raw_doc["title"]} {raw_doc["abstract"]}"

    corpus.append(text)
    doc_ids.append(raw_doc["id"])

# -- 3. Build Document-Term Matirx
print(f"Building the document-term matrix from {MAX_DOC} documents")

vectorizer = TfidfVectorizer(stop_words="english")

document_term_matrix = vectorizer.fit_transform(corpus)

# -- 4. Save the vectorizer, document-term matrix and the doc_ids list
with open("./pickles/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("./pickles/document_term_matrix.pkl", 'wb') as f:
    pickle.dump(document_term_matrix, f)

with open("./pickles/doc_ids.pkl", "wb") as f:
    pickle.dump(doc_ids, f)

print("Building document-term matrix completed")


