import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Function to preprocess text NR+SR+LC
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

# Downsampling the happy class
happy_songs = data[data['mood'] == 'happy']
if len(happy_songs) > 150:
    np.random.seed(42)
    drop_indices = np.random.choice(happy_songs.index, size=90, replace=False)
    data = data.drop(drop_indices)
    print('Downsampling successful!')
    print(f'Number of songs in each mood after downsampling:\n{data["mood"].value_counts()}')

data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(data['processed_lyrics'], data['mood'], test_size=0.2, random_state=42, stratify=data['mood'])

from sklearn.model_selection import GridSearchCV

pipeline_tfidf = make_pipeline(
    TfidfVectorizer(),
    KNeighborsClassifier()
)

pipeline_bow = make_pipeline(
    CountVectorizer(),
    KNeighborsClassifier()
)

param_grid_tfidf = {
    'tfidfvectorizer__max_df': [0.5, 0.75, 1.0],
    'tfidfvectorizer__min_df': [1, 2, 3],
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
    'kneighborsclassifier__n_neighbors': [3, 5, 11, 19, 25, 31,41],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__metric': ['euclidean', 'manhattan']
}

param_grid_bow = {
    'countvectorizer__max_df': [0.5, 0.75, 1.0],
    'countvectorizer__min_df': [1, 2, 3],
    'countvectorizer__ngram_range': [(1, 1), (1, 2)],
    'kneighborsclassifier__n_neighbors': [3, 5, 11, 19, 25, 31,41],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__metric': ['euclidean', 'manhattan']
}

grid_search_tfidf = GridSearchCV(pipeline_tfidf, param_grid_tfidf, cv=5, verbose=1, n_jobs=-1)
grid_search_tfidf.fit(X_train, y_train)
print("Best parameters for TF-IDF:")
print(grid_search_tfidf.best_params_)
y_pred_tfidf = grid_search_tfidf.predict(X_test)
print("Classification Report for TF-IDF with Grid Search:")
print(classification_report(y_test, y_pred_tfidf))
print(f"Accuracy with TF-IDF: {accuracy_score(y_test, y_pred_tfidf)}")

grid_search_bow = GridSearchCV(pipeline_bow, param_grid_bow, cv=5, verbose=1, n_jobs=-1)
grid_search_bow.fit(X_train, y_train)
print("Best parameters for BoW:")
print(grid_search_bow.best_params_)
y_pred_bow = grid_search_bow.predict(X_test)
print("Classification Report for BoW with Grid Search:")
print(classification_report(y_test, y_pred_bow))
print(f"Accuracy with BoW: {accuracy_score(y_test, y_pred_bow)}")


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

plot_confusion_matrix(y_test, y_pred_bow, "Confusion Matrix for BoW")
plot_confusion_matrix(y_test, y_pred_tfidf, "Confusion Matrix for TF-IDF")


def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
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

plot_learning_curve(grid_search_bow.best_estimator_, X_train, y_train)
plot_learning_curve(grid_search_tfidf.best_estimator_, X_train, y_train)
