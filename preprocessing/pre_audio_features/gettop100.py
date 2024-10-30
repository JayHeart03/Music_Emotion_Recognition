import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Spotify API credentials
client_id = "5366f9af81dc4ed1b5a024581249e45d"
client_secret = "0122b3bd64e34c81935cfb78ee6acec1"

# Initialize Spotipy with the client credentials
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Get the playlist data
playlist_id = "6unJBM7ZGitZYFJKkO0e4P"

playlist_data = sp.playlist(playlist_id)

# Extract track information and audio features
columns = ['track_id', 'artist', 'track_name', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
track_features_df = pd.DataFrame(columns=columns)

# Loop through each track in the playlist
for track in playlist_data["tracks"]["items"]:
    track_info = track["track"]
    track_id = track_info["id"]
    artist_name = track_info["artists"][0]["name"]
    track_name = track_info["name"]
    track_duration = track_info["duration_ms"]

    audio_features = sp.audio_features(track_id)[0]

    new_row_df = pd.DataFrame([[track_id, artist_name, track_name, track_duration, audio_features['danceability'], audio_features['energy'], audio_features['key'], audio_features['loudness'], audio_features['mode'], audio_features['speechiness'], audio_features['acousticness'], audio_features['instrumentalness'], audio_features['liveness'], audio_features['valence'], audio_features['tempo']]], columns=columns)

    track_features_df = pd.concat([track_features_df, new_row_df], ignore_index=True)

track_features_df.to_csv('data/data_top100/top_hits_of_2023_audio_features.csv', index=False)

print("CSV file saved successfully.")
