import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data_path = 'data/MoodyLyrics.csv'
data = pd.read_csv(data_path)

continuous_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                       'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

scaler = StandardScaler()
data[continuous_features] = scaler.fit_transform(data[continuous_features])

for feature in continuous_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='mood', y=feature)
    plt.title(f'Boxplot of {feature} by Mood (Standardized)')
    plt.xlabel('Mood')
    plt.ylabel('Standardized Value')
    plt.show()
