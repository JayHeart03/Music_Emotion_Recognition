import pandas as pd
import numpy as np
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical,pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, InputLayer
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import itertools

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

# Downsampling 90 'happy' songs
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

valid_combinations = []
# Generate all possible combinations of preprocessing options
for combination in itertools.product([True, False], repeat=len(preprocess_options)):
    params = dict(zip(preprocess_options, combination))
    if params['Lemma'] and params['Stem']:
        continue
    valid_combinations.append(params)

# Preprocess the text data with each valid combination of options
for preprocess_params in valid_combinations:
    preprocess_param_str = ', '.join([k for k, v in preprocess_params.items() if v])
    print(f"Running with options: {preprocess_param_str}")

    data['processed_lyrics'] = data['lyrics'].apply(lambda x: preprocess_text(x, **preprocess_params))
    
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(data['processed_lyrics'])
    sequences = tokenizer.texts_to_sequences(data['processed_lyrics'])
    data_pad = pad_sequences(sequences, maxlen=250) # padding the sequences to a maximum length of 250

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data['mood'])
    y = to_categorical(integer_encoded)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_pad, y, test_size=0.2, random_state=42, stratify=y)

    from gensim.models import KeyedVectors
    # Load the Word2Vec model
    word2vec_path = 'data/GoogleNews-vectors-negative300.bin'
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    embedding_dim = 300  
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if i >= vocab_size:
            continue
        try:
            embedding_vector = word2vec[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except KeyError:
            pass  
    
    # Define the BiLSTM model
    model = Sequential([
        InputLayer(input_shape=(250,)),
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=250, trainable=False),
        Dropout(0.2),  
        Bidirectional(LSTM(100)), 
        Dense(4, activation='softmax',activity_regularizer=l2(0.001)) 
    ])

    learning_rate = 0.0006

    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    epochs = 25 
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.1)

    loss, accuracy = model.evaluate(X_test, y_test)
    for layer in model.layers:
        weights = layer.get_weights()
        print(f"Layer: {layer.name}")
        for i, weight in enumerate(weights):
            print(f"Weight {i}: shape {weight.shape}")

    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

    # Make predictions and calculate the F1 score
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    f1 = f1_score(y_true, y_pred_classes, average="weighted")

    results_data.append({
        'Preprocessing_Combination': preprocess_param_str,
        'Accuracy': accuracy,
        'F1_Score': f1
    })

results = pd.DataFrame(results_data)
results.to_csv('bilstm_preprocessing_results.csv', index=False)

