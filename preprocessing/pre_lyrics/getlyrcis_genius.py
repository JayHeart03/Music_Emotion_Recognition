import lyricsgenius
import pandas as pd

# GENIUS_API get lyrics
GENIUS_API_TOKEN = 'pZifG4hwqHLJtlXrYrMdsPJSdt06iInUxrgVmxsU4NvbeAz-I85OrLfOloUzWfv3'

genius = lyricsgenius.Genius(GENIUS_API_TOKEN)

df = pd.read_excel('data/data_moody/MoodyLyrics_Orignal.xlsx', skiprows=14, header=1)

lyrics_data = []
for index, row in df.iterrows():
    try:
        song = genius.search_song(row['Title'], row['Artist'])
        lyrics = song.lyrics.replace('\n', ' ') if song else None
    except Exception as e:
        print(f"Error occurred while getting lyrics for {row['Artist']} - {row['Title']}: {e}")
        lyrics = None
    lyrics_data.append({
        'ML_Index': row['Index'],
        'artist': row['Artist'],
        'title': row['Title'],
        'mood': row['Mood'],
        'lyrics': lyrics
    })

lyrics_df = pd.DataFrame(lyrics_data)

lyrics_df.to_csv('data/data_moody/MoodyLyrics_get.csv', index=False)
