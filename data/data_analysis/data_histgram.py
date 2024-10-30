import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

data = pd.read_csv('data/MoodyLyrics.csv')

# Define the continuous features that we are interested in
continuous_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

# Define a preprocessor using the ColumnTransformer.
# This will apply StandardScaler to the continuous features.
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), continuous_features)
])

processed_data = preprocessor.fit_transform(data)
# Create a new DataFrame with the scaled data
processed_df = pd.DataFrame(processed_data, columns=[*continuous_features])

# Calculate the number of rows and columns needed for the subplots
total_features = len(processed_df.columns)
cols = 4  # You can choose an appropriate number of columns
rows = math.ceil(total_features / cols)

plt.figure(figsize=(20, 15))
for i, feature in enumerate(processed_df.columns, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(processed_df[feature], kde=False, bins=15)
    plt.title(feature)

plt.tight_layout()
plt.suptitle('Overall Feature Distributions', y=1.02)
plt.show()
