import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from umap import UMAP

# Reading the dataset
data = pd.read_csv('data/data_moody/MoodyLyrics.csv')

# Downsampling the 'happy' mood to balance the dataset
# happy_songs = data[data['mood'] == 'happy']

# if len(happy_songs) > 150:
#     np.random.seed(42)
#     drop_indices = np.random.choice(happy_songs.index, size=90, replace=False)
#     data = data.drop(drop_indices)
#     print('Downsampling successful!')
#     print(f'Number of songs in each mood after downsampling:\n{data["mood"].value_counts()}')

# Feature columns in the dataset
feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                   'speechiness', 'acousticness', 'instrumentalness', 
                   'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

# Calculating feature ranges for analysis
feature_ranges = data[feature_columns].agg(['min', 'max']).transpose()

# Selecting continuous features for transformation
continuous_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                       'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

# Creating a pipeline for scaling continuous features
continuous_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Setting up a Column Transformer for the continuous features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_features),
    ])

# Preprocessing the data
processed_data = preprocessor.fit_transform(data)

# Applying PCA, t-SNE, and UMAP for dimensionality reduction to 2D and 3D
pca_2d = PCA(n_components=2)
pca_result_2d = pca_2d.fit_transform(processed_data)

tsne_2d = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
tsne_result_2d = tsne_2d.fit_transform(processed_data)

# umap_model_2d = UMAP(n_components=2)
# umap_result_2d = umap_model_2d.fit_transform(processed_data)

pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(processed_data)

tsne_3d = TSNE(n_components=3, verbose=1, perplexity=50, n_iter=1000)
tsne_result_3d = tsne_3d.fit_transform(processed_data)

# umap_model_3d = UMAP(n_components=3)
# umap_result_3d = umap_model_3d.fit_transform(processed_data)

# Creating DataFrames for the dimensionality reduced data
df_pca_2d = pd.DataFrame(data=pca_result_2d, columns=['PCA1', 'PCA2'])
df_tsne_2d = pd.DataFrame(data=tsne_result_2d, columns=['tSNE1', 'tSNE2'])
# df_umap_2d = pd.DataFrame(data=umap_result_2d, columns=['UMAP1', 'UMAP2'])

df_pca_3d = pd.DataFrame(data=pca_result_3d, columns=['PCA1', 'PCA2', 'PCA3'])
df_tsne_3d = pd.DataFrame(data=tsne_result_3d, columns=['tSNE1', 'tSNE2', 'tSNE3'])
# df_umap_3d = pd.DataFrame(data=umap_result_3d, columns=['UMAP1', 'UMAP2', 'UMAP3'])

# Adding mood labels to the new DataFrames
mood_label = data['mood']
df_pca_2d['mood'] = df_tsne_2d['mood'] = mood_label
df_pca_3d['mood'] = df_tsne_3d['mood'] = mood_label

# Plotting the 2D and 3D representations using PCA, t-SNE, and UMAP
plt.figure(figsize=(16, 7))
# [Code for plotting 2D PCA and t-SNE by Mood]

plt.figure(figsize=(10, 6))
# [Code for plotting 2D UMAP by Mood]

fig = plt.figure(figsize=(10, 8))
# [Code for plotting 3D PCA by Mood]

fig = plt.figure(figsize=(10, 8))
# [Code for plotting 3D t-SNE by Mood]

fig = plt.figure(figsize=(10, 8))
# [Code for plotting 3D UMAP by Mood]
