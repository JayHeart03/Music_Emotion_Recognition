import pandas as pd
import matplotlib.pyplot as plt

file_path = 'data/data_moody/MoodyLyrics.csv'
data = pd.read_csv(file_path)

# Dropping rows with missing values in the specified columns
data_cleaned = data.dropna(subset=['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature'])

cleaned_file_path = 'data/data_moody/MoodyLyrics_audio_cleaned.csv'
data_cleaned.to_csv(cleaned_file_path, index=False)

# Displaying the distribution of moods in the dataset
total_entries = data_cleaned.shape[0]

mood_distribution = data_cleaned['mood'].value_counts(normalize=True) * 100

plt.figure(figsize=(10, 6))
mood_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribution of Moods in Songs')
plt.xlabel('Mood')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

print(f"Total complete entries: {total_entries}")
print("Mood distribution in the dataset (in %):")
print(mood_distribution)
