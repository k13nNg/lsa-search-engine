#  Latent Semantic Analysis on arXiv papers
![CleanShot 2025-12-02 at 00 11 32](https://github.com/user-attachments/assets/6a7f6567-2cc9-4806-b3c1-e222bb0795a5)

# Abstract

This project implements a semantic search engine, built from first principle, using **Latent Semantic Analysis (LSA)**. Unlike keyword-based systems, this engine retrieves documents based on their *conceptual similarity,* leveraging the power of Linear Algebra.

The system indexed a corpus of 50000 arXiv papers in the following categories:

- Artificial Intelligence,
- Cryptography,
- Machine Learning,
- Numerical Analysis,
- Combinatorics,
- Logic, and
- Optimization

The engine then constructs a sparse Document-Term matrix, and performs Truncated Singular Value Decomposition (SVD) to extract the top $k=100$ singular values (a.k.a semantic concepts).

The engine was able to uncover hidden connections between various mathematical concepts (papers on similar concepts are grouped close together), without explicit instructions.

# Mathematical Foundations

The core of the search engine is the ***Singular Value Decomposition (SVD) Theorem***, which states:

<!-- <img width="2048" height="806" alt="image" src="https://github.com/user-attachments/assets/c40828ca-4dc3-4e27-a325-ace400054940" /> -->

>** Theorem: (Singular Value Decomposition of Matrices)**
>
>Let $A \in M_{m \times n}(\mathbb{F})$ be a matrix of rank $r$ with non-zero singular values $\sigma_1 \geq \cdots \geq \sigma_r > 0$.
>Then there exist unitary matrices $U \in M_{m \times m}(\mathbb{F})$ and $V \in M_{n \times n}(\mathbb{F})$ such that
>
>$$
>A = U \Sigma V^*,
>$$
>
>where $\Sigma$ is the $m \times n$ matrix whose entries are
>
>$$
\Sigma_{ij} = \begin{cases}
\sigma_i & \text{if } i = j \leq r \\
0 & \text{otherwise.}
\end{cases}
$$
>
>If $A \in M_{m \times n}(\mathbb{R})$ is real, then $U$ and $V$ can be chosen to be orthogonal matrices.


Since computing and storing the full SVD of a matrix poses significant challenges in practice (and the fact that we can express $\Sigma$ as just a square matrix by dropping the extra zero rows), the LSA uses the $\text{rank-}k$ truncation of the matrix $A$, which only computes $k$ largest singular values, for some $k \in \mathbb{Z}_{\geq 0}$. 

<!-- <img width="1796" height="358" alt="image" src="https://github.com/user-attachments/assets/bcdb4d94-1fad-4b90-b20f-a2bc8dfd2384" /> -->

>**Definition: (Rank-k truncation)**
>
>Let $A \in M_{m \times n}(\mathbb{F})$ be a rank $r$ matrix with singular values $\sigma_1 \geq \cdots \geq \sigma_r > 0$ and compact singular value decomposition $A = U_r \Sigma_r V_r^*$, where $U = [\vec{u}_1 \cdots \vec{u}_r]$ and $V = [\vec{v}_1 \cdots \vec{v}_r]$.
>Let $k \leq r$ be a positive integer. The **rank-k truncation** of $A$ is
>
>$$
A_k = \sigma_1 \vec{u}_1 \vec{v}_1^* + \cdots + \sigma_k \vec{u}_k \vec{v}_k^*.
$$
>
>
In this version of SVD, $U$ would be an $m\times k$ matrix, $\Sigma$ would be a $k \times k$ matrix, and $V^T$ would be a $k \times n$ matrix. 

By the ***Eckart-Young Theorem**,* the $\text{rank-}k$ truncation of the matrix $A$, denoted $A_k$ (obtained by the Truncated SVD algorithm), is the closest $\text{rank-}k$ matrix approximation of $A$. Thus, with the constraints on computing power and storage, we can compute the Truncated SVD instead of the Full SVD of the matrix and still obtain reasonably good approximations.

<!-- <img width="1796" height="380" alt="image" src="https://github.com/user-attachments/assets/5d939c7d-b27d-48fb-859a-7358780d2dc2" /> -->

>**Theorem: (Eckart–Young Theorem)**
>
> Let $A \in M_{m \times n}(\mathbb{F})$, and let $A_k$ be the rank-$k$ truncation of $A$. Let $B \in M_{m \times n}(\mathbb{F})$ be an arbitrary rank $k$ matrix. Then
>
> $$
> \|A - B\| \geq \|A - A_k\|.
> $$

(Source https://www.math.uwaterloo.ca/~f2alfais/notes/math235-notes.pdf).

# Parameter
<p align="center">
<img width="80%" height="1138" alt="image" src="https://github.com/user-attachments/assets/8f15bbe0-509c-4bdb-b541-aa7b0b2d6958" />
</p>

Visual inspection of the scree plots reveals a significant drop in singular values for $k=10$ and $k=50$, indicating that these distinct rank approximations likely discard meaningful structure. However, for $k=100$ and above, the curve flattens significantly, particularly beyond the 60th component. This plateau suggests that dimensions beyond this point offer diminishing returns in capturing semantic information. Hence, $k=100$ was selected as the optimal trade-off between computational efficiency and information retention.

# Methodology

Overall, the search engine can be decomposed into two parts: Indexing, and Searching.

## Indexing

Given a list of documents, the engine first constructs a [document-term matrix](https://en.wikipedia.org/wiki/Document-term_matrix) $A$ using the TF-IDF function (so it can capture the importance of word occurrences with respect to each document). Note that $A$ is an $m \times n$ matrix, where $m$ is the number of documents, and $n$ is the number of distinct terms.

Since the TF-IDF function always return a real value, $U$ and $V$ are orthogonal matrices (From the above ***Theorem***). This implies $V^* = V^{-1} = V^T$.

Then, the engine applies the truncated SVD algorithm, with $k = 100$, to decompose the document-term matrix $A$ into the following product:

$$
\begin{align*}
	A &= U \Sigma V^{\*} = U \Sigma V^{T} \quad (\text{from above})
\end{align*}
$$

where

- $U$ is an $m \times k$ matrix
- $\Sigma$ is a $k \times k$ matrix
- $V^T$ is a $k \times n$ matrix

The matrices $U$, $\Sigma$ and $V^T$ can be interpreted as follows:

- $U$ is the document-concept matrix. The higher the entry $U_{ij}$ is, the more relevant document $i$ is to concept $j$, vice versa.
- $\Sigma$ is the concept-concept matrix. Note that this is a diagonal matrix. The higher the entry $\Sigma_{ii}$ is, the more distinct concept $i$ is among the other concepts covered by the papers.
- $V^T$ is the concept-term matrix. The higher the entry $V^T_{ij}$ is, the more closely related concept $i$ is to term $j$.

Upon generation, the $U$, $\Sigma$ and $V^T$ matrices are serialized into `.pkl` format. This ensures data persistence, allowing the model to be efficiently reloaded in the searching phase without re-computation.

## Searching

The system processes an input query $q$ by encoding it into a binary vector, $q_{vec}$. In this representation, the $i$-th entry is set to $1$ if the corresponding vocabulary term appears in the query, and $0$ otherwise. The engine then maps this sparse vector into the latent concept space via the operation:

$$
\begin{align*}
q_{concept} &= q_{vec} \cdot (V^T)^T = q_{vec} \cdot V
\end{align*}
$$

Intuitively, since $q_{vec}$ represents the query in "term space" and $V$ acts as a term-to-concept mapping matrix, their product yields the query's coordinates within the concept space.

Next, the engine generates the document representations by computing:

$$
\begin{align*}
	P = U \cdot \Sigma
\end{align*}
$$

This operation projects the entire document corpus onto the concept space. 

Consequently, each row vector of matrix $P$ represents a specific document's position in this reduced-dimensional space. Finally, with both the query ($q_{concept}$) and the documents ($P$) projected onto the same basis, the system computes the cosine similarity between $q_{concept}$ and every row in $P$. The engine then retrieves and returns the IDs of the documents with the highest similarity scores.

# Note

Although the visualization script only shows points in 3D space (which only contains 3 most dominant concepts), the search for articles uses all 100 dimensions for improved semantic accuracy.

# How to run the search engine

1. Clone the repo:
    
    ```bash
    git clone https://github.com/k13nNg/lsa-search-engine.git
    cd lsa-search-engine
    ```
    
2. Install dependencies
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. Download the [arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv), place the JSON file in the folder `./data`
4. Run the `data_filter.py` file. This script filters the JSON file to only contain the relevant papers (specified in the ***Abstract*** section).
5. Run the `preprocessing.py`  script. This script creates a document-term matrix from the papers, and the TFIDF vectorizer `pickle`  image, which is needed for encoding query later.
6. Run the `decomposer.py` script. This script runs the Truncated SVD algorithm on the document-term matrix and save the $U$, $\Sigma$ and $V^T$ in `.pkl`  format for the search engine to use later.
7. Run the `visualizer.py` script. This script build a dash app that runs at `http://127.0.0.1:8050/`
