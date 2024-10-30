import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score,f1_score
from collections import Counter
import numpy as np

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
# Shuffle the data
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

print(data[['lyrics', 'processed_lyrics']].head())
word_count = sum(len(text.split()) for text in data['processed_lyrics'])
print(f"Total number of words in the processed lyrics: {word_count}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['processed_lyrics'], data['mood'], test_size=0.20, random_state=42, stratify=data['mood'])

# Create a pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()), 
    ('nb', MultinomialNB())
])

# Define the parameters for the grid search
parameters = {
    'bow__ngram_range': [(1, 2), (1, 3)],
    'bow__max_df': [0.5, 0.75, 1.0],
    'bow__min_df': [1,2,3],
    'nb__alpha': [0.1,0.3,0.5]
}

# Perform the grid search
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5)
print("Performing grid search...")
grid_search.fit(X_train, y_train)
print("Grid search done!")
print(f"Best score: {grid_search.best_score_}")\

best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print(f"{param_name}: {best_parameters[param_name]}")

# Evaluate the model
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"F1 score: {f1_score(y_test, y_pred, average='weighted')}")


# Print the top 10 features per class
print("Top 10 features per class:")
feature_names = grid_search.best_estimator_.named_steps['bow'].get_feature_names_out()
for i, class_name in enumerate(grid_search.best_estimator_.classes_):
    top10 = np.argsort(grid_search.best_estimator_.named_steps['nb'].feature_log_prob_[i])[-10:]
    print("%s: %s" % (class_name, ", ".join(feature_names[j] for j in top10)))

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
train_sizes, train_scores, test_scores = learning_curve(grid_search.best_estimator_, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")
plt.legend(loc="best")
plt.show()
