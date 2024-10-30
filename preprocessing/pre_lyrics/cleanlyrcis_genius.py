import pandas as pd
import matplotlib.pyplot as plt
import re
import unidecode
from langdetect import detect, LangDetectException

file_path = 'data/data_moody/MoodyLyrics.csv' # after getting the lyrics from the API
data_df = pd.read_csv(file_path)

data_df = data_df.dropna(subset=['lyrics'])
data_df = data_df[(data_df['lyrics'].str.len() > 100) & (data_df['lyrics'].str.len() < 5000)]

# Remove lyrics phrases that are not part of the song
def clean_lyrics(text):
    phrases_to_remove = [
        r'\[.*?\]', 
        r'\d+ Contributors', 
        r'EmbedShare URLCopyEmbedCopy', 
        r'1 Contributor',
        r'\d*Embed$',          
        r'^.*?Lyrics',       
        r'You might also like'
    ]
    for phrase in phrases_to_remove:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    return text.strip()

data_df['lyrics'] = data_df['lyrics'].apply(clean_lyrics)

data_df = data_df.drop_duplicates(subset=['artist', 'title'])
# Remove non-English songs
def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

data_df = data_df[data_df['lyrics'].apply(is_english)]

def normalize_text(text):
    return unidecode.unidecode(text).replace("’", "'").replace("‘", "'").lower()

# Remove songs where the title is not present in the lyrics
def is_title_present_in_lyrics(row):
    title_words = normalize_text(row['title']).split()
    lyrics = normalize_text(row['lyrics'])
    return any(word in lyrics for word in title_words)
data_df = data_df[data_df.apply(is_title_present_in_lyrics, axis=1)]

mood_counts = data_df['mood'].value_counts()
plt.figure(figsize=(10, 6))
mood_counts.plot(kind='bar', color='teal')
plt.title('Number of Songs per Mood Label')
plt.xlabel('Mood Label')
plt.ylabel('Number of Songs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

output_file = 'data/data_moody/MoodyLyrics_cleand.csv'
data_df.to_csv(output_file, index=False)

print("Data processed and visualized. Output saved to", output_file)