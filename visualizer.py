import plotly.graph_objects as go
from engine import search, get_doc_points
from dash import Dash, dcc, html, Input, Output, State

# only plot 10000 points so we don't overwhelm the browser
LIMIT_POINTS = 10000

doc_vecs, doc_ids= get_doc_points(LIMIT_POINTS)

x_coords = doc_vecs[:, 0]
y_coords = doc_vecs[:, 1]
z_coords = doc_vecs[:, 2]

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Latent Semantic Analysis 3D Visualization", style={'textAlign': 'center'}),

    html.H3([
        "Trained on ",
        html.A("ArXiV data", 
               href="https://www.kaggle.com/datasets/Cornell-University/arxiv", # Replace with your GitHub URL if you prefer
               target="_blank", # Opens in new tab
               style={'color': '#007BFF', 'textDecoration': 'underline'}),
        html.Span(" | ", style={'color': '#ccc', 'margin': '0 10px'}),
        html.A("GitHub Repo", 
               href="YOUR_GITHUB_URL_HERE",
               target="_blank",
               style={'color': '#007BFF', 'textDecoration': 'underline'})
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),

    html.Div([
        html.Div([
            html.H3("Search"),
            dcc.Input(
                id="search-box", type="text", placeholder="Enter a topic (e.g. \"neural networks\")",
                style={'width': '100%', 'padding': '10px', 'boxSizing': 'border-box'}
            ),
            html.Button(
                'Search', id='btn-search', n_clicks=0,
                style={'width': '100%', 'padding': '10px', 'marginTop': '10px', 
                       'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'cursor': 'pointer'}
            ),
            html.Hr(),
                    
            html.Div(id='search-results', style={'overflowY': 'auto', 'maxHeight': '60vh'})

        ], style={'width': '25%', 'padding': '20px', 'borderRight': '1px solid solid #ddd'}),

        html.Div([
            html.H2("Latent 3D Search Space", style={'textAlign': 'center'}),
            dcc.Graph(id='3d-scatter', style={'height': '75vh', 'width': '100%'})
        ], style={'width': '100%'})
    ], style={'display': 'flex', 'height': '85vh'}),

])

@app.callback(
    [Output('3d-scatter', 'figure'),
     Output('search-results', 'children')],
    [Input('btn-search', 'n_clicks')],
    [State('search-box', 'value')]
)

def update_view(n_clicks, query_text):
    results_html = [html.P("Enter a query to see related papers.")]

    docs= go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode="markers",
        name='Documents',
        text=[f"arXiv ID: {id}" for id in doc_ids],
        marker= dict(size=3,
                    color=z_coords,           
                    colorscale='Viridis', 
                    opacity=1)
    )
    layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3',
                bgcolor='#f4f4f4'
            ),
            legend=dict(x=0.05, y=0.95)
        )
    data = [docs]

    if (query_text):
        search_results = search(query_text)
        results_html = []

        if (search_results):
            for result in search_results[0]:
                rank, score, doc_id = result
                
                # Create a clickable card for each result
                card = html.Div([
                    html.Strong(f"{rank}. arXiv:{doc_id}"),
                    html.Span(f" (Score: {score:.2f})", style={'color': '#666', 'fontSize': '0.9em'}),
                    html.Br(),
                    html.A("View on arXiv", href=f"https://arxiv.org/abs/{doc_id}", target="_blank", 
                            style={'fontSize': '0.8em', 'color': '#007BFF'})
                ], style={'padding': '10px', 'borderBottom': '1px solid #eee', 'backgroundColor': 'white'})
                
                results_html.append(card)

            q_x, q_y, q_z = search_results[1]

            trace_query = go.Scatter3d(
                    x=[q_x], y=[q_y], z=[q_z],
                    mode='markers+text',
                    name='Query', text=[query_text],
                    marker=dict(size=15, color='red', symbol='diamond')
                )
            
            data.append(trace_query)

        else:
            results_html = [html.Div("⚠️ Unknown term.", style={'color': 'red', 'fontWeight': 'bold'})]

    return go.Figure(data=data, layout=layout), results_html

if __name__ == '__main__':
    app.run(debug=True)