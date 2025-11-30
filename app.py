import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pickle
from scipy.sparse import diags
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION & LOADING ---
PICKLE_PATH = "./pickles/"
LIMIT_POINTS = 10000 
TOP_K = 5

print("Loading Engine Artifacts...")
try:
    with open(f"{PICKLE_PATH}U_matrix.pkl", "rb") as f:
        U = pickle.load(f)
    with open(f"{PICKLE_PATH}s_matrix.pkl", "rb") as f:
        s = pickle.load(f)
    with open(f"{PICKLE_PATH}Vt_matrix.pkl", "rb") as f:
        Vt = pickle.load(f)
    with open(f"{PICKLE_PATH}vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{PICKLE_PATH}doc_ids.pkl", "rb") as f:
        doc_ids = pickle.load(f)
    print("Artifacts loaded.")
except Exception as e:
    print(f"Error: {e}")
    exit()

# --- 2. DATA PREPARATION ---
# Full Document Vectors for Search (Search against ALL papers, not just the plotted ones)
doc_vectors = U @ diags(s)

# Subsample for Visualization (Only plot 10k to keep browser fast)
vis_data = doc_vectors[:LIMIT_POINTS]
vis_ids = doc_ids[:LIMIT_POINTS]

x_coords = vis_data[:, 0]
y_coords = vis_data[:, 1]
z_coords = vis_data[:, 2]

# --- 3. DASH LAYOUT ---
app = dash.Dash(__name__)

app.layout = html.Div([
    # Title Bar
    html.Div([
        html.H2("Spectral Search Engine", style={'margin': '0', 'color': 'white'}),
        html.P("Latent Semantic Analysis (SVD) on arXiv", style={'margin': '0', 'color': '#ddd'}),
    ], style={'backgroundColor': '#222', 'padding': '20px', 'textAlign': 'center'}),

    # Main Container (Flexbox for Side-by-Side)
    html.Div([
        
        # LEFT PANEL: Search & Results
        html.Div([
            html.H3("Search", style={'marginTop': '0'}),
            dcc.Input(
                id='search-box', type='text', placeholder='e.g. "optimal control"',
                style={'width': '100%', 'padding': '10px', 'boxSizing': 'border-box'}
            ),
            html.Button(
                'Search', id='btn-search', n_clicks=0,
                style={'width': '100%', 'padding': '10px', 'marginTop': '10px', 
                       'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'cursor': 'pointer'}
            ),
            
            html.Hr(),
            
            html.Div(id='search-results', style={'overflowY': 'auto', 'maxHeight': '60vh'})
            
        ], style={'width': '25%', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRight': '1px solid #ddd'}),

        # RIGHT PANEL: 3D Visualization
        html.Div([
            dcc.Graph(id='3d-scatter', style={'height': '85vh'})
        ], style={'width': '75%', 'padding': '0'}),
        
    ], style={'display': 'flex', 'height': '85vh'})
])

# --- 4. CALLBACKS ---
@app.callback(
    [Output('3d-scatter', 'figure'),
     Output('search-results', 'children')],
    [Input('btn-search', 'n_clicks')],
    [State('search-box', 'value')]
)
def update_view(n_clicks, query_text):
    # Default: Empty results, Base Plot
    results_html = [html.P("Enter a query to see related papers.", style={'color': '#888'})]
    
    # 1. Base Plot (The Galaxy)
    trace_docs = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        name='Documents',
        text=vis_ids,
        marker=dict(size=3, color=x_coords, colorscale='Viridis', opacity=0.5)
    )
    data = [trace_docs]
    
    # 2. Handle Search
    if query_text:
        # A. Vectorize
        q_vec = vectorizer.transform([query_text])
        
        if q_vec.nnz == 0:
            results_html = [html.Div("⚠️ Unknown term.", style={'color': 'red', 'fontWeight': 'bold'})]
        else:
            # B. Project (The Math)
            q_concept = q_vec @ Vt.T
            
            # C. Search (Cosine Similarity against ALL docs)
            # Compare (1, k) against (N, k)
            scores = cosine_similarity(q_concept, doc_vectors).flatten()
            
            # Get Top K indices
            top_indices = scores.argsort()[::-1][:TOP_K]
            
            # D. Build Result List (HTML)
            results_html = []
            results_html.append(html.H4(f"Top {TOP_K} Matches:"))
            
            for rank, idx in enumerate(top_indices):
                score = scores[idx]
                doc_id = doc_ids[idx]
                
                # Create a clickable card for each result
                card = html.Div([
                    html.Strong(f"{rank+1}. arXiv:{doc_id}"),
                    html.Span(f" (Score: {score:.2f})", style={'color': '#666', 'fontSize': '0.9em'}),
                    html.Br(),
                    html.A("View on arXiv", href=f"https://arxiv.org/abs/{doc_id}", target="_blank", 
                           style={'fontSize': '0.8em', 'color': '#007BFF'})
                ], style={'padding': '10px', 'borderBottom': '1px solid #eee', 'backgroundColor': 'white'})
                
                results_html.append(card)

            # E. Add Query Marker to Plot
            qx, qy, qz = q_concept[0, 0], q_concept[0, 1], q_concept[0, 2]
            trace_query = go.Scatter3d(
                x=[qx], y=[qy], z=[qz],
                mode='markers+text',
                name='Query', text=[query_text],
                marker=dict(size=15, color='red', symbol='diamond')
            )
            data.append(trace_query)

    # 3. Layout Styling
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3',
            bgcolor='#f4f4f4'
        ),
        legend=dict(x=0.05, y=0.95)
    )

    return go.Figure(data=data, layout=layout), results_html

if __name__ == '__main__':
    app.run(debug=True)