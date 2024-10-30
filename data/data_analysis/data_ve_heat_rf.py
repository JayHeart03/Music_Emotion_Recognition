import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('data/MoodyLyrics.csv')
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

happy_songs = data[data['mood'] == 'happy']

continuous_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                       'instrumentalness', 'valence', 'tempo', 'duration_ms', 'loudness', 'liveness']
categorical_features = ['key', 'mode', 'time_signature']

# Creating transformers for continuous and categorical data
continuous_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  
])

# Preprocessing data using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Applying the transformations to the dataset
processed_data = preprocessor.fit_transform(data)

# Constructing column names for the processed data
processed_columns = (continuous_features + list(preprocessor.named_transformers_['cat']
                          .named_steps['onehot']
                          .get_feature_names_out(categorical_features)))

processed_df = pd.DataFrame(processed_data, columns=processed_columns)

processed_data_with_mood = pd.concat([processed_df, data['mood'].reset_index(drop=True)], axis=1)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='valence', y='energy', hue='mood', data=processed_data_with_mood)
plt.xlabel('Valence')
plt.ylabel('Energy')
plt.title('Scatter Plot of Valence vs. Energy by Mood')
plt.legend(title='Mood', loc='upper right')
plt.grid(True)
plt.show()

# Encoding the mood labels for correlation analysis
mood_encoder = OneHotEncoder()
mood_encoded = mood_encoder.fit_transform(data[['mood']]).toarray()
mood_feature_names = mood_encoder.get_feature_names_out(['mood'])
mood_encoded_df = pd.DataFrame(mood_encoded, columns=mood_feature_names)

# Combining processed data with encoded mood labels
processed_data_with_mood_encoded = pd.concat([processed_df, mood_encoded_df], axis=1)

# Generating and plotting correlation heatmap
correlation_matrix = processed_data_with_mood_encoded.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap between Features and Encoded Mood')
plt.show()

# Splitting the dataset for training and testing
X = processed_df 
y = data['mood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Defining the model and Recursive Feature Elimination
rf_model = RandomForestClassifier()
rfe = RFE(rf_model, n_features_to_select=30)
rfe.fit(X_train, y_train)

# Selecting features as per RFE
selected_features = np.array(processed_df.columns)[rfe.support_]

# Fitting the RandomForest model with selected features
rf_model.fit(X_train[selected_features], y_train)

# Extracting and sorting feature importances
importances = rf_model.feature_importances_
feature_importances = list(zip(selected_features, importances))
feature_importances_sorted = sorted(feature_importances, key=lambda x: x[1], reverse=True)

print("Feature importances in descending order:")
for feature, importance in feature_importances_sorted:
    print(f"{feature}: {importance}")

features_sorted, importances_sorted = zip(*feature_importances_sorted)
plt.figure(figsize=(15, 10))
plt.barh(features_sorted, importances_sorted, color='skyblue')
plt.xlabel('Relative Importance')
plt.title('Sorted Feature Importances in RandomForest Classifier with RFE')
plt.gca().invert_yaxis()
plt.show()

print(classification_report(y_test, rf_model.predict(X_test[selected_features])))
print("Accuracy:", rf_model.score(X_test[selected_features], y_test))
