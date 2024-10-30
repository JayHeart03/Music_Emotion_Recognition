import pandas as pd
import numpy as np
import itertools
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score,f1_score

data_path = 'data/data_moody/MoodyLyrics.csv'

data = pd.read_csv(data_path)
# Randomly shuffle the data and reset the index
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Downsample the 'happy' mood class
happy_songs = data[data['mood'] == 'happy']
if len(happy_songs) > 150:
    np.random.seed(42)
    drop_indices = np.random.choice(happy_songs.index, size=90, replace=False)
    data = data.drop(drop_indices)
    print('Downsampling successful!')
    print(f'Number of songs in each mood after downsampling:\n{data["mood"].value_counts()}')

# Define the text preprocessing functions
def remove_noise(text):
    return ' '.join([word for word in word_tokenize(text) if word.isalpha()])

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in word_tokenize(text) if word.lower() not in stop_words])

def to_lowercase(text):
    return text.lower()

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])

def stem(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in word_tokenize(text)])


# Combine the preprocessing functions
def preprocess_text(text, NR=False, SR=False, LC=False, Lemma=False, Stem=False):
    if NR:
        text = remove_noise(text)
    if SR:
        text = remove_stopwords(text)
    if LC:
        text = to_lowercase(text)
    if Lemma and not Stem:
        text = lemmatize(text)
    elif Stem and not Lemma:
        text = stem(text)
    return text

preprocess_options = ['NR', 'SR', 'LC', 'Lemma', 'Stem']

results_data = []

valid_combinations = []
# Generate all possible combinations of preprocessing options
for combination in itertools.product([True, False], repeat=len(preprocess_options)):
    params = dict(zip(preprocess_options, combination))
    if params['Lemma'] and params['Stem']:
        continue
    valid_combinations.append(params)

# Run the model with all valid combinations of preprocessing options
for preprocess_params in valid_combinations:
    preprocess_param_str = ', '.join([k for k, v in preprocess_params.items() if v])
    # print(f"Running with options: {preprocess_params}")
    print(f"Running with options: {preprocess_params}")

    data['processed_lyrics'] = data['lyrics'].apply(lambda x: preprocess_text(x, **preprocess_params))

    tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=0.001, ngram_range=(1, 1))
    X = tfidf_vectorizer.fit_transform(data['processed_lyrics'])
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['mood'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    svc = SVC(kernel='linear', C=0.36)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("\n") 

    results_data.append({
        'Preprocessing_Combination': preprocess_param_str,
        'Accuracy': accuracy,
        'F1_Score': f1
    })
results = pd.DataFrame(results_data)

results.to_csv('preprocessing_results.csv', index=False)