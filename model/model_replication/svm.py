import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve

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

data = data.dropna(subset=['lyrics', 'mood'])
data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)

# Load the GloVe embeddings
def load_glove_embeddings(path):
    embeddings_index = {}
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_embeddings = load_glove_embeddings('data/data_moody/glove.6B.100d.txt') 
def text_to_glove(text, embeddings_index):
    tokens = word_tokenize(text.lower())
    text_embedding = np.mean([embeddings_index.get(word, np.zeros(100)) for word in tokens], axis=0)
    return text_embedding if text_embedding.size else np.zeros(100)

# Downsample the 'happy' mood class
happy_songs = data[data['mood'] == 'happy']

if len(happy_songs) > 150:
    np.random.seed(42)
    drop_indices = np.random.choice(happy_songs.index, size=90, replace=False)
    data = data.drop(drop_indices)
    print('Downsampling successful!')
    print(f'Number of songs in each mood after downsampling:\n{data["mood"].value_counts()}')

# Preprocess the lyrics
word_count = sum(len(text.split()) for text in data['processed_lyrics'])
print(data[['lyrics', 'processed_lyrics']].head())
print(f"Total number of words in the processed lyrics: {word_count}")

# Transform the text to GloVe vectors
X_glove = np.array([text_to_glove(text, glove_embeddings) for text in data['processed_lyrics']])

from sklearn.preprocessing import LabelEncoder
# Encode the mood labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['mood'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_glove, y, test_size=0.2, random_state=42,stratify=y)

# Train the SVM model
svm_model = SVC(kernel='linear', C=1, random_state=42)

svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)

classification_report_result = classification_report(y_test, y_pred)
accuracy_score_result = accuracy_score(y_test, y_pred)

print("Classification Report:\n", classification_report_result)
print("Accuracy:", accuracy_score_result)
print("f1_score:", f1_score(y_test, y_pred, average='weighted'))

# Plot the learning curve and validation curve
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

def plot_validation_curve(estimator, title, X, y, param_name, param_range, axes=None, ylim=None, cv=None,
                        n_jobs=None):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(8, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel(f"{param_name}")
    axes.set_ylabel("Score")

    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring="accuracy", n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes.grid()
    axes.fill_between(param_range, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(param_range, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(param_range, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(param_range, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

cv = 10
n_jobs = -1

plot_learning_curve(svm_model, "Learning Curve (SVM)", X_train, y_train, axes=axes[0], ylim=(0, 1.0), cv=cv, n_jobs=n_jobs, train_sizes=np.linspace(.1, 1.0, 10))
param_range = [0.1, 0.35, 1, 10, 100]
plot_validation_curve(svm_model, "Validation Curve (SVM)",  X_train, y_train, param_name="C", param_range=param_range, axes=axes[1], ylim=(0, 1.0), cv=cv, n_jobs=n_jobs)

plt.show()
