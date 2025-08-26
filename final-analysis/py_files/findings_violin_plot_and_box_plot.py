import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
# Plotly - Data Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chart_studio
import chart_studio.plotly as py
import plotly.express as px

# chart-studio username and API key
chart_studio.tools.set_credentials_file(username='mandy_huang', api_key='KmKfCFT8gHvsXAAPqXr7')

def violin_plot_for_mood_features(df):

    # Filter the dataset for pop_label equal to 2;1;0, meaning the songs are most popular
    df_label_2 = df[df['pop_label'] == 2]

    # Define the mood features
    features = ['danceability', 'energy', 'valence', 'tempo']

    # Define subplot titles
    subplot_titles = [f'Violin Plot of {feature.capitalize()}' for feature in features]

    # Create a subplot figure
    fig = make_subplots(rows=1, cols=4, subplot_titles=subplot_titles)

    # Add a violin plot to the figure for each feature
    for i, feature in enumerate(features, 1):
        fig.add_trace(
            go.Violin(y=df_label_2[feature], name=feature.capitalize(),
                      box_visible=True, meanline_visible=True),
            row=1, col=i
        )

    # Update layout for each violin plot
    fig.update_traces(orientation='v')
    fig.update_layout(
        title_text='Violin Plots for Songs with Popularity Label 2 (most popular)',
        height=600,
        width=1200
    )

    # Show the figure
    fig.show()

    py.plot(fig, filename='Violin Plots for Songs with Popularity Label 2 (most popular)', auto_open=True)


def box_plot_for_mood_features(df):
    # Filter the dataframe for pop_label=2
    pop_label_2_df = df[df['pop_label'] == 2]

    # Specified features to plot
    features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'liveness', 'valence', 'tempo',
        'time_signature', 'duration_min'
    ]

    # Define a set of colors
    colors = px.colors.qualitative.Plotly

    # Create a plot with multiple sub-plots
    rows = 3  # Number of rows
    cols = 4  # Number of columns, adjusted to 4 for 10 features
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=features)

    # Add a box plot for each audio feature
    for i, feature in enumerate(features, start=1):
        row = (i - 1) // cols + 1
        col = (i - 1) % cols + 1
        color = colors[i % len(colors)]  # Cycle through the defined colors
        fig.add_trace(
            go.Box(y=pop_label_2_df[feature], x=pop_label_2_df['predicted_mood'], name=feature, marker_color=color),
            row=row, col=col
        )

    # Generate the plot
    fig.update_layout(height=800, width=1300, title_text="Audio Features of Spotify popular songs by Predicted Mood Labels")
    fig.show()

    py.plot(fig, filename='Audio Features of Spotify popular songs by Predicted Mood Labels', auto_open=True)




def main():

    # import spotify_popular_songs dataset
    df_result = pd.read_csv('../csv_files/spotify_popular_songs.csv')

    # Draw violin plot for 4 mood features
    violin_plot_for_mood_features(df_result)

    # import spotify_popular_songs_mood_predicted dataset
    data_predicted = pd.read_csv('../csv_files/spotify_popular_songs_mood_predicted.csv')

    # Draw box plot for audio features classified by mood of Spotify popular songs
    box_plot_for_mood_features(data_predicted)

    return

if '__main__' == __name__:
    main()