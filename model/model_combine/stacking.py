import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


# Download the necessary resources for NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text NR+SR+LC
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
#shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Downsample the 'happy' mood class
happy_songs = data[data['mood'] == 'happy']

if len(happy_songs) > 150:
    np.random.seed(42)
    drop_indices = np.random.choice(happy_songs.index, size=90, replace=False)
    data = data.drop(drop_indices)
    print('Downsampling successful!')
    print(f'Number of songs in each mood after downsampling:\n{data["mood"].value_counts()}')

# Preprocess the lyrics
data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)

# Define continuous features
continuous_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                       'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

# Data preprocessing for continuous features
continuous_transformer = Pipeline(steps=[('scaler', StandardScaler())])
preprocessor = ColumnTransformer(transformers=[('num', continuous_transformer, continuous_features)])

# Encode the mood labels
label_encoder = LabelEncoder()
data['mood_encoded'] = label_encoder.fit_transform(data['mood'])

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['mood_encoded'])

# Create a TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_df=0.5, min_df=0.001, ngram_range=(1, 1))
X_train_text = tfidf.fit_transform(train_data['processed_lyrics']).toarray()
X_test_text = tfidf.transform(test_data['processed_lyrics']).toarray()

audio_features_train = preprocessor.fit_transform(train_data)
audio_features_test = preprocessor.transform(test_data)

# Combine the text and audio features
X_train_combined = np.hstack((X_train_text, audio_features_train))
X_test_combined = np.hstack((X_test_text, audio_features_test))

# Encode the labels
train_labels = train_data['mood_encoded']
test_labels = test_data['mood_encoded']

# svm text
svm_model = SVC(kernel='linear', C=0.36, probability=True)

# rf audio
rf_model = RandomForestClassifier(n_estimators=100)
print(rf_model.get_params())

# kfold cross-validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)

meta_features_train = []
meta_features_test = []

# Stacking
for model in [svm_model, rf_model]:
    meta_train = cross_val_predict(model, X_train_combined, train_labels, cv=kf, method="predict_proba")
    meta_features_train.append(meta_train)
    
    model.fit(X_train_combined, train_labels)
    meta_test = model.predict_proba(X_test_combined)
    meta_features_test.append(meta_test)

meta_features_train = np.concatenate(meta_features_train, axis=1)
meta_features_test = np.concatenate(meta_features_test, axis=1)

# Train the meta-classifier
meta_model = XGBClassifier(n_estimators=100, learning_rate=0.4, max_depth=3, objective='multi:softmax')

meta_model.fit(meta_features_train, train_labels)

y_pred_meta = meta_model.predict(meta_features_test)
print("Accuracy:", accuracy_score(test_labels, y_pred_meta))
print(classification_report(test_labels, y_pred_meta))
print(confusion_matrix(test_labels, y_pred_meta))

# Plot the learning curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title="Learning Curve", ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

print("Stacking Classifier Cross-Validation Score:", np.mean(cross_val_score(meta_model, meta_features_train, train_labels, cv=kf)))
plot_learning_curve(meta_model, meta_features_train, train_labels, cv=kf)
plt.show()

# Save the models
# import joblib
# joblib.dump(meta_model, 'meta_model.joblib')
# joblib.dump(tfidf, 'tfidf.joblib')

