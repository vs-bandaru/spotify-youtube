import pandas as pd
import numpy as np

# Plotly - Data Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chart_studio
import chart_studio.plotly as py
import plotly.express as px

# chart-studio username and API key
chart_studio.tools.set_credentials_file(username='mandy_huang', api_key='KmKfCFT8gHvsXAAPqXr7')

def histogram_correlation(df):

    ###########################
    ##### Histograms
    ###########################

    # Get all attributes name for spotify_api_song dataset
    attributes_1 = ['danceability','energy','key','loudness','mode','speechiness','acousticness','intrumentalness','liveness']
    attributes_2 = ['valence','tempo','time_signature','track_popularity','artist_popularity','duration_min']

    # Specify a color for each subgraph
    colors_1 = ['#a7c957', '#d62728', '#386641', '#99621e', '#ff7f0e', '#bc4749',
              '#dd7373', '#2d3047', '#ff9da7']
    colors_2 = ['#59a14f', '#af7aa1', '#bab0ac', '#edc949', '#4e79a7', '#9c755f', '#f28e2b', '#17becf', '#8c271e', '#6a994e']

    # Create a subgraph grid with 6 rows and 3 columns
    fig = make_subplots(rows=3, cols=3, subplot_titles=attributes_1)

    # Adds a histogram to each subgraph
    for i, attribute in enumerate(attributes_1):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.add_trace(
            go.Histogram(x=df[attribute], marker_color=colors_1[i], nbinsx=25,hovertemplate="<b>Bin Edges:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"),
            row=row, col=col
        )

    fig.update_layout(height=600, width=900, title_text="Frequency Distributions")
    fig.update_traces(showlegend=False)

    fig.show()

    py.plot(fig, filename='histogram_1', auto_open=True)

    # Create a subgraph grid with 6 rows and 3 columns
    fig = make_subplots(rows=2, cols=3, subplot_titles=attributes_2)

    # Adds a histogram to each subgraph
    for i, attribute in enumerate(attributes_2):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.add_trace(
            go.Histogram(x=df[attribute], marker_color=colors_2[i], nbinsx=25,
                         hovertemplate="<b>Bin Edges:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"),
            row=row, col=col
        )

    fig.update_layout(height=600, width=800, title_text="Frequency Distributions")
    fig.update_traces(showlegend=False)

    fig.show()

    py.plot(fig, filename='histogram_2', auto_open=True)

    ###########################
    ##### Correlations graph
    ###########################

    # Calculate the correlation matrix
    corr = df.corr()

    # Create a heatmap
    fig = px.imshow(corr,
                    labels=dict(x="Attributes", y="Attributes", color="Correlation"),
                    x=df.columns,
                    y=df.columns,
                    color_continuous_scale='RdBu_r')

    # Update the layout for a better look
    fig.update_layout(title='Attributes Correlation Heatmap',
                      margin=dict(l=20, r=20, t=30, b=20))

    # Show the plot
    fig.show()

    py.plot(fig, filename='correlations_plot', auto_open=True)


def main():

    # import spotify_api_song_cleaned dataset
    df_song = pd.read_csv('../csv_files/spotify_api_song_cleaned.csv')
    df_song_cleaned = df_song.drop(columns=['track_id'])

    # Use plotly to generate the histogram and correlation for features in spotify_api_song_cleaned dataset
    histogram_correlation(df_song_cleaned)

if '__main__' == __name__:
    main()
