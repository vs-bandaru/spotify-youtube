import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import chart_studio
import chart_studio.plotly as py
import plotly.express as px

df = pd.read_csv('../csv_files/popularity_prediction_result.csv')

# Initialize Dash application
app = dash.Dash(__name__)

# Heatmap data
data = {
    "track_popularity": df['track_popularity'],
    "artist_popularity": df['artist_popularity'],
    "likes": df['like_count'],
    "views": df['view_count'],
    "sentiment_score": df['senti_score']
}
df_heatmap = pd.DataFrame(data)
corr_matrix = df_heatmap.corr()

# Generate heatmap
fig_heatmap = px.imshow(
    corr_matrix,
    labels=dict(x="Feature", y="Feature", color="Correlation"),
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    text_auto=True,
    title="Heatmap of Feature Correlations"
)

# Set application layout
app.layout = html.Div([
    dcc.RadioItems(
        id='popularity-selector',
        options=[
            {'label': 'All', 'value': 'all'},
            {'label': 'Non-Popular (0)', 'value': '0'},
            {'label': 'Moderately Popular (1)', 'value': '1'},
            {'label': 'Very Popular (2)', 'value': '2'}
        ],
        value='all',
        labelStyle={'display': 'block'}
    ),
    dcc.Graph(
        id='youtube_popularity-graph',
        style={'height': '800px', 'width': '80%'}
    ),
    dcc.Graph(
        id='spotify_popularity-graph',
        style={'height': '800px', 'width': '80%'}
    ),
    dcc.Graph(
        id='feature-correlation-heatmap',
        style={'height': '800px', 'width': '80%'},
        figure=fig_heatmap
    )
])

# Define callback to update layout
@app.callback(
    Output('youtube_popularity-graph', 'figure'),
    [Input('popularity-selector', 'value')]
)
def update_graph_1(selected_popularity):
    if selected_popularity == 'all':
        # All points are grey
        color = 'grey'
    elif selected_popularity == '0':
        # popularity label 0 are green
        color = np.where(df['pop_label'] == 0, 'green', 'grey')
    elif selected_popularity == '1':
        # popularity label 1 are blue
        color = np.where(df['pop_label'] == 1, 'blue', 'grey')
    else:
        # popularity label 2 are red
        color = np.where(df['pop_label'] == 2, 'red', 'grey')

    # Generate youtube popularity graph
    fig = go.Figure(data=[go.Scatter3d(
        x=np.log1p(df['like_count']),
        y=np.log1p(df['view_count']),
        z=df['senti_score'],
        text=[f'track_popularity: {a}<br>artist_popularity: {b}' for a, b in zip(df['track_popularity'], df['artist_popularity'])],
        hoverinfo='x+y+z+text',
        mode='markers',
        marker=dict(size=5, color=color, opacity=0.5)
    )])

    fig.update_layout(
        title='Likes_Views_Sentiment score',
        scene=dict(
            xaxis_title='Likes',
            yaxis_title='Views',
            zaxis_title='Sentiment Score'
        )
    )

    return fig


@app.callback(
    Output('spotify_popularity-graph', 'figure'),
    [Input('popularity-selector', 'value')]
)
def update_graph_2(selected_popularity):
    if selected_popularity == 'all':
        # All points are grey
        color = 'grey'
    elif selected_popularity == '0':
        # popularity label 0 are green
        color = np.where(df['pop_label'] == 0, 'green', 'grey')
    elif selected_popularity == '1':
        # popularity label 1 are blue
        color = np.where(df['pop_label'] == 1, 'blue', 'grey')
    else:
        # popularity label 2 are red
        color = np.where(df['pop_label'] == 2, 'red', 'grey')

    # Generate Spotify popularity graph
    fig = go.Figure(data=[go.Scatter(
        x=df['track_popularity'],
        y=df['artist_popularity'],
        mode='markers',
        marker=dict(size=5, color=color, opacity=0.5)
    )])

    fig.update_layout(
        title='track_popularity and artist_popularity',
        xaxis_title='track_popularity',
        yaxis_title='artist_popularity',

    )

    return fig


def main():
    app.run_server(debug=True)


# Run application
if __name__ == '__main__':
    main()