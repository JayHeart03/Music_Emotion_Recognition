import lyricsgenius
import pandas as pd

# GENIUS_API get lyrics
GENIUS_API_TOKEN = 'ITvPM8bhBxPwgagGR8f7SlE7pyJLCuLSwW-VQYd5KxiqAYne40CDlx1U0hdiHaPr'

genius = lyricsgenius.Genius(GENIUS_API_TOKEN)

df = pd.read_csv('data/data_moody/MoodyLyrics.csv')

lyrics_data = []
for index, row in df.iterrows():
    try:
        song = genius.search_song(row['title'], row['artist'])
        lyrics = song.lyrics.replace('\n', ' ') if song else None
    except Exception as e:
        print(f"Error occurred while getting lyrics for {row['artist']} - {row['title']}: {e}")
        lyrics = None
    lyrics_data.append({
        'ML_Index': row['ML_Index'], 
        'artist': row['artist'],
        'title': row['title'],
        'mood': row['mood'],
        'lyrics': lyrics
    })

lyrics_df = pd.DataFrame(lyrics_data)

lyrics_df.to_csv('mergedsad_lyrics_data.csv', index=False)
