import pandas as pd
import re

file_path = 'data/data_moody/MoodyLyrics.csv'
df = pd.read_csv(file_path)

# Dropping rows with missing values in the specified columns
df['lyrics'] = df['lyrics'].apply(lambda x: re.sub(r"\[.*?\]", "", str(x)).replace("\n", " "))

# Removing special characters (like apostrophes) from 'Artist' and 'Title'
excluded_texts = ['Incorrect Song Information or Artist', 'Lyrics Not Found', 'Lyrics Not Found or Timeout']
df = df[~df['lyrics'].isin(excluded_texts)]

# Removing non-English songs
df['is_english'] = df['lyrics'].apply(lambda x: x.isascii())
df = df[df['is_english']]

df = df.drop('is_english', axis=1)

final_csv_path = 'data/cleaned_lyrics_and_features_data.csv'
df.to_csv(final_csv_path, index=False)
