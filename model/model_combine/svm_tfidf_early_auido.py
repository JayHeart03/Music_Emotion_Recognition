import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.svm import SVC

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define the text preprocessing function LC+NR+SR
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
#shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Downsample the 'happy' mood class
data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)

happy_songs = data[data['mood'] == 'happy']

if len(happy_songs) > 150:
    np.random.seed(42)  
    drop_indices = np.random.choice(happy_songs.index, size=90, replace=False)
    data = data.drop(drop_indices)
    print('Downsampling successful!')
    print(f'Number of songs in each mood after downsampling:\n{data["mood"].value_counts()}')

# Define continuous features
continuous_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
# categorical_features = ['key', 'mode', 'time_signature']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), continuous_features),
    # ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X = data['processed_lyrics']
y = data['mood']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Preprocess the text data
text_transformer = TfidfVectorizer(max_df=0.5, min_df=0.001, ngram_range=(1, 1)) 
X_train_text = text_transformer.fit_transform(X_train)
X_test_text = text_transformer.transform(X_test)

audio_features_train = preprocessor.fit_transform(data.loc[X_train.index])
audio_features_test = preprocessor.transform(data.loc[X_test.index])

# Combine text and audio features
X_train_combined = np.hstack((X_train_text.toarray(), audio_features_train))
X_test_combined = np.hstack((X_test_text.toarray(), audio_features_test))

# SVM text
svm_text = SVC(kernel='linear', C=0.36)
svm_text .fit(X_train_text, y_train)
y_pred_text = svm_text.predict(X_test_text)
print("Performance with Text Features:")
print(classification_report(y_test, y_pred_text))
print(f"Accuracy: {accuracy_score(y_test, y_pred_text)}\n")

# SVM combined
svm_combined = SVC(kernel='linear',C=0.36)
svm_combined.fit(X_train_combined, y_train)
y_pred_combined = svm_combined.predict(X_test_combined)
print("Performance with Combined Features:")
print(classification_report(y_test, y_pred_combined))
print(f"Accuracy: {accuracy_score(y_test, y_pred_combined)}")

from sklearn.model_selection import cross_val_score

# Cross-validation
svm_cross_val_scores = cross_val_score(svm_combined, X_train_combined, y_train, cv=5)
svm_cross_val_scores2 = cross_val_score(svm_text, X_train_text, y_train, cv=5)
print("Average Cross-validation Score2:", np.mean(svm_cross_val_scores2))
print("Average Cross-validation Score:", np.mean(svm_cross_val_scores))


# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
# train_sizes, train_scores, test_scores = learning_curve(svm_combined, X_train_combined, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), verbose=0)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
# plt.figure()
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
# plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
# plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
# # print("best score: ", svm_combined.best_score_)
# plt.xlabel('Training examples')
# plt.ylabel('Score')
# plt.title('Learning curve')
# plt.legend(loc='best')
# plt.show()

# print("best score: ", svm_combined.best_score_)
# print("best params: ", svm_combined.best_params_)

#save the models and transformers
# import joblib
# joblib.dump(svm_combined, 'svm_tfidf_early_audio.joblib')
# # joblib.dump(svm_text, 'svm_tfidf_early_text.joblib')

# joblib.dump(text_transformer, 'tfidf_vectorizer_early.joblib')



