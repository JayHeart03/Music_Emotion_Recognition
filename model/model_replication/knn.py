import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score,f1_score
from sklearn.preprocessing import FunctionTransformer

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

# Function to load GloVe model
def load_glove_model(glove_file_path):
    print("Loading Glove Model")
    model = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
    print(f"{len(model)} words loaded!")
    return model

# Function to transform text to GloVe vectors
def glove_transform(texts, glove_model):
    dim = len(next(iter(glove_model.values())))
    return np.array([
        np.mean([glove_model.get(word, np.zeros(dim)) for word in text.split()], axis=0)
        for text in texts
    ])
# Load the dataset
data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
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

word_count = sum(len(text.split()) for text in data['processed_lyrics'])
print(data[['lyrics', 'processed_lyrics']].head())
print(f"Total number of words in the processed lyrics: {word_count}")

glove_path = 'data/data_moody/glove.6B.100d.txt'
glove_model = load_glove_model(glove_path)

# Initialize the pipeline
glove_transformer = FunctionTransformer(glove_transform, kw_args={'glove_model': glove_model})

# Define the pipeline
pipeline = make_pipeline(
    glove_transformer,
    KNeighborsClassifier(n_neighbors=29)
)

# Fit the pipeline
knn_classifier = pipeline.named_steps['kneighborsclassifier']
print("p parameter value:", knn_classifier.p)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['processed_lyrics'], data['mood'], test_size=0.2, random_state=42, stratify=data['mood'])

pipeline.fit(X_train, y_train)
# Make predictions
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
