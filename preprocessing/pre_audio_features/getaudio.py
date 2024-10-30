import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import os

# Spotify API credentials
client_id = '30a6df1a0f7040ccb5c3c05f49aade03'
client_secret = '906fcdb460be42ce9422305783f44e97'

# Initialize Spotipy with the client credentials
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, retries=5)

df = pd.read_csv('data/data_moody/MoodyLyrics.csv')
output_file = 'data/data_moody/MoodyLyrics_lyrcis_cleaned_feature.csv'
request_delay = 3

# Create a copy of the original dataframe without the features
df_copy = df.copy()

# Ensure the output file is empty or does not exist before starting
if os.path.exists(output_file):
    os.remove(output_file)

for index, row in df.iterrows():
    print(f"Processing {index+1}/{len(df)}: {row['artist']} - {row['title']}")
    success = False
    retries = 0
    while not success and retries < 5:
        try:
            time.sleep(request_delay) # Delay to avoid hitting the rate limit
            
            # Search for the track on Spotify
            results = sp.search(q='artist:' + row['artist'] + ' track:' + row['title'], type='track', limit=1)
            items = results['tracks']['items']
            if items:
                track = items[0]
                features = sp.audio_features(track['id'])[0]
                print(f"Features found for {row['artist']} - {row['title']}")
            else:
                features = {}
                print(f"No features found for {row['artist']} - {row['title']}")

            # Update the copy of the dataframe with the new features
            for key, value in features.items():
                df_copy.at[index, key] = value

            # Save the updated dataframe to CSV
            df_copy.to_csv(output_file, index=False)
            success = True
        
        # Handle rate limit and other exceptions
        except spotipy.SpotifyException as e:
            if e.http_status == 429:
                wait_time = int(e.headers.get('Retry-After', 10)) 
                print(f"Rate limit reached, waiting for {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"Error processing {row['artist']} - {row['title']}: {e}")
                break
        except Exception as e:
            print(f"Unhandled exception processing {row['artist']} - {row['title']}: {e}")
            break

print("All songs processed.")
