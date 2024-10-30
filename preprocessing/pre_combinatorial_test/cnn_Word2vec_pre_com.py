import pandas as pd
import numpy as np
import itertools
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical, pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import gensim
from keras.callbacks import LearningRateScheduler

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

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


data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
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
    
# Define the preprocessing options
preprocess_options = ['NR', 'SR', 'LC', 'Lemma', 'Stem']
results_data = []

# Generate all possible combinations of preprocessing options
valid_combinations = []
for combination in itertools.product([True, False], repeat=len(preprocess_options)):
    params = dict(zip(preprocess_options, combination))
    if params['Lemma'] and params['Stem']:
        continue
    valid_combinations.append(params)

# Define the learning rate scheduler
def exponential_decay(epoch):
    initial_lr = 0.0005 
    k = 0.1 
    lr = initial_lr * np.exp(-k * epoch)
    return lr

# Define the learning rate scheduler
lr_scheduler = LearningRateScheduler(exponential_decay)

# Run the model with all valid combinations of preprocessing options
for preprocess_params in valid_combinations:
    preprocess_param_str = ', '.join([k for k, v in preprocess_params.items() if v])
    print(f"Running with options: {preprocess_param_str}")

    data['processed_lyrics'] = data['lyrics'].apply(lambda x: preprocess_text(x, **preprocess_params))

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['mood'])
    labels = to_categorical(encoded_labels)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['processed_lyrics'])
    sequences = tokenizer.texts_to_sequences(data['processed_lyrics'])

    max_sequence_length = 250
    data_tensor = pad_sequences(sequences, maxlen=max_sequence_length)

    x_train, x_val, y_train, y_val = train_test_split(data_tensor, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Load the pre-trained Word2Vec model
    word2vec_path = 'data/GoogleNews-vectors-negative300.bin'
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    # Prepare the embedding matrix
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 300
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in word2vec_model.key_to_index:
            embedding_vector = word2vec_model[word]
            embedding_matrix[i] = embedding_vector
    
    # Define the CNN model
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, embeddings_initializer=Constant(embedding_matrix), input_length=max_sequence_length, trainable=False))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.2))
    model.add(Conv1D(32, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.2))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(len(labels[0]), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val),callbacks=[lr_scheduler], verbose=1)

    # Evaluate the model
    scores = model.evaluate(x_val, y_val, verbose=1)
    accuracy = scores[1]
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    f1 = f1_score(y_true, y_pred_classes, average="weighted")

    results_data.append({
        'Preprocessing_Combination': preprocess_param_str,
        'Accuracy': accuracy,
        'F1_Score': f1
    })

results = pd.DataFrame(results_data)
results.to_csv('cnn3_preprocessing_results.csv', index=False)
