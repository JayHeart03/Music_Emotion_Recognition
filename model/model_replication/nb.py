import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score,f1_score
from collections import Counter
import numpy as np

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Function to preprocess text Lemma+NR+SR+LC
def preprocess_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in lemmatized if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
# Downsample the 'happy' mood class
data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)

happy_songs = data[data['mood'] == 'happy']
# Downsample the 'happy' mood class
if len(happy_songs) > 150:
    np.random.seed(42) 
    drop_indices = np.random.choice(happy_songs.index, size=90, replace=False)
    data = data.drop(drop_indices)
    print('Downsampling successful!')
    print(f'Number of songs in each mood after downsampling:\n{data["mood"].value_counts()}')

word_count = sum(len(text.split()) for text in data['processed_lyrics'])
print(data[['lyrics', 'processed_lyrics']].head())
print(f"Total number of words in the processed lyrics: {word_count}")

# Find the top word for each mood
top_words_per_mood = {}
for mood in data['mood'].unique():
    lyrics_mood = data[data['mood'] == mood]['processed_lyrics']
    word_counts = Counter(" ".join(lyrics_mood).split())
    top_word, count = word_counts.most_common(1)[0]
    top_words_per_mood[mood] = (top_word, count)

top_words_df = pd.DataFrame(top_words_per_mood.items(), columns=['Mood', 'Top Word (Count)'])
print(top_words_df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['processed_lyrics'], data['mood'], test_size=0.20, random_state=42, stratify=data['mood'])

# Train a Naive Bayes classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

parameters = {
    'nb__alpha': [0.05] # Original paper used alpha=0.05
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5)
grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print(f"{param_name}: {best_parameters[param_name]}")

# Evaluate the model
y_pred = grid_search.predict(X_test)

print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")

print("Top 10 features per class:")
feature_names = grid_search.best_estimator_.named_steps['tfidf'].get_feature_names_out()
for i, class_name in enumerate(grid_search.best_estimator_.classes_):
    top10 = np.argsort(grid_search.best_estimator_.named_steps['nb'].feature_log_prob_[i])[-10:]
    print("%s: %s" % (class_name, ", ".join(feature_names[j] for j in top10)))


