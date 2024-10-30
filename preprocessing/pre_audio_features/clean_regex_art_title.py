import pandas as pd

file_path_3 = 'data/data_moody/MoodyLyrics.csv'
df_3 = pd.read_csv(file_path_3)

# Removing special characters (like apostrophes) from 'Artist' and 'Title'
df_3['artist'] = df_3['artist'].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
df_3['title'] = df_3['title'].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)

cleaned_file_path_3 = 'data/data_moody/MoodyLyrics_art_tit_cleaned.csv'
df_3.to_csv(cleaned_file_path_3, index=False)
