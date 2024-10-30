import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define the text preprocessing function Lemma+LC+NR+SR
def preprocess_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in lemmatized if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
#shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)

# Downsample the 'happy' mood class
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
    ('num', MinMaxScaler(), continuous_features),
    # ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X = data['processed_lyrics']
y = data['mood']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Preprocess the text data
text_transformer = CountVectorizer(max_df=0.5, min_df=3, ngram_range=(1, 2))
X_train_text = text_transformer.fit_transform(X_train)
X_test_text = text_transformer.transform(X_test)

# Preprocess the audio features
audio_features_train = preprocessor.fit_transform(data.loc[X_train.index])
audio_features_test = preprocessor.transform(data.loc[X_test.index])

# Combine text and audio features
X_train_combined = np.hstack((X_train_text.toarray(), audio_features_train))
X_test_combined = np.hstack((X_test_text.toarray(), audio_features_test))

# Train the Naive Bayes model with text features
nb_text = MultinomialNB(alpha=0.5)
nb_text.fit(X_train_text, y_train)
y_pred_text = nb_text.predict(X_test_text)
print("Performance with Text Features:")
print(classification_report(y_test, y_pred_text))
print(f"Accuracy: {accuracy_score(y_test, y_pred_text)}\n")

# Train the Naive Bayes model with combined features
nb_combined = MultinomialNB(alpha=0.5)
nb_combined.fit(X_train_combined, y_train)
y_pred_combined = nb_combined.predict(X_test_combined)
print("Performance with Combined Features:")
print(classification_report(y_test, y_pred_combined))
print(f"Accuracy: {accuracy_score(y_test, y_pred_combined)}")

# Calculate the number of out-of-vocabulary words
test_texts = X_test.apply(preprocess_text)
unique_words_in_test = set(word for text in test_texts for word in text.split())
vocabulary = set(text_transformer.vocabulary_.keys())
oov_words = unique_words_in_test - vocabulary
num_oov_words = len(oov_words)
print(f"Number of out-of-vocabulary words in test set: {num_oov_words}")


# Save the models and transformers
# import joblib
# joblib.dump(nb_combined, 'nb_combined_bow.joblib')

# joblib.dump(text_transformer, 'nb_countvectorizer.joblib')

# joblib.dump(nb_text, 'nb_text_bow.joblib')









