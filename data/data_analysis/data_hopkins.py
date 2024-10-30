import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

data = pd.read_csv('data/MoodyLyrics.csv')
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Define the continuous features for preprocessing
continuous_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), continuous_features),
    # ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

def hopkins_statistic(X, subsample=0.1, seed=42):
    n = X.shape[0]  # Number of samples
    d = X.shape[1]  # Number of features
    m = int(subsample * n)  # Size of random sample set

    np.random.seed(seed)
    # Initialize NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2).fit(X)

    # Generate uniform random points (u points) in the same space as the data
    rand_X = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(m, d))
    u, _ = nbrs.kneighbors(rand_X, return_distance=True)
    u = u[:, 0]  # Get distances to nearest points

    # Generate random points (w points) from the data samples themselves
    idx = np.random.choice(n, size=m, replace=False)
    w, _ = nbrs.kneighbors(X[idx], n_neighbors=2, return_distance=True)
    w = w[:, 1]  # Get distances to second nearest points

    # Calculate the mean distances
    U = np.mean(u)
    W = np.mean(w)
    # Calculate the Hopkins statistic
    H = U / (U + W)

    return H

# Compute Hopkins statistic for each mood category
moods = data['mood'].unique()
hopkins_values = {}
for mood in moods:
    mood_data = data[data['mood'] == mood]
    X_preprocessed = preprocessor.fit_transform(mood_data[continuous_features])
    hopkins_values[mood] = hopkins_statistic(X_preprocessed)

print(hopkins_values)
