# Import necessary libraries
import pandas as pd
import numpy as np
import gensim
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, Input, concatenate
from keras.initializers import Constant
from keras.utils import to_categorical, pad_sequences
from keras.callbacks import LearningRateScheduler

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pickle

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define the text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

# Load and preprocess the dataset
data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# Downsample the 'happy' mood class
happy_songs = data[data['mood'] == 'happy']

if len(happy_songs) > 150:
    np.random.seed(42)
    drop_indices = np.random.choice(happy_songs.index, size=90, replace=False)
    data = data.drop(drop_indices)
    print('Downsampling successful!')
    print(f'Number of songs in each mood after downsampling:\n{data["mood"].value_counts()}')


data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)

# Define continuous features
continuous_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'loudness']

# Data preprocessing for continuous features
continuous_transformer = Pipeline(steps=[('scaler',  StandardScaler())])
preprocessor = ColumnTransformer(transformers=[('num', continuous_transformer, continuous_features)])
audio_features = data[continuous_features]
processed_audio_features = preprocessor.fit_transform(audio_features)

# Tokenize and pad the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['processed_lyrics'])
sequences = tokenizer.texts_to_sequences(data['processed_lyrics'])
max_sequence_length = 250
data_padded = pad_sequences(sequences, maxlen=max_sequence_length)

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['mood'])
labels = to_categorical(encoded_labels)

# Load the Word2Vec model
word2vec_path = 'data/data_moody/GoogleNews-vectors-negative300.bin'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# Prepare the embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 300
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in word2vec_model.key_to_index:
        embedding_vector = word2vec_model[word]
        embedding_matrix[i] = embedding_vector

# Define the neural network architecture
def create_model():
    # Define the neural network architecture
    text_input = Input(shape=(max_sequence_length,), dtype='int32', name='text_input')
    embedding_layer = Embedding(vocab_size, embedding_dim, embeddings_initializer=Constant(embedding_matrix), input_length=max_sequence_length, trainable=False)(text_input)
    conv1 = Conv1D(128, 5, activation='relu')(embedding_layer)
    maxpool1 = MaxPooling1D(5)(conv1)
    dropout1 = Dropout(0.2)(maxpool1)
    conv2 = Conv1D(64, 5, activation='relu')(dropout1)
    maxpool2 = MaxPooling1D(5)(conv2)
    dropout2 = Dropout(0.2)(maxpool2)
    conv3 = Conv1D(32, 5, activation='relu')(dropout2)
    dropout3 = Dropout(0.2)(conv3)
    global_maxpool = GlobalMaxPooling1D()(dropout3)

    audio_input = Input(shape=(audio_features.shape[1],), name='audio_input')
    dense_audio = Dense(64, activation='relu')(audio_input)
    dropout_audio = Dropout(0.2)(dense_audio)
    dense_audio = Dense(32, activation='relu')(dropout_audio)
    dropout_audio = Dropout(0.2)(dense_audio)
    dense_audio = Dense(16, activation='relu')(dropout_audio)

    merged = concatenate([global_maxpool,dense_audio])
    classifier = Dense(len(labels[0]), activation='softmax')(merged)

    model = Model(inputs=[text_input, audio_input], outputs=classifier) 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
# Define a learning rate scheduler
def exponential_decay(epoch):
    initial_lr = 0.0005
    k = 0.1
    lr = initial_lr * np.exp(-k * epoch)
    return lr

lr_scheduler = LearningRateScheduler(exponential_decay)

# Define the StratifiedKFold parameters
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Prepare the data
X_text = data_padded
X_audio = processed_audio_features
Y = encoded_labels

# Start the cross-validation process
cv_scores = []

for fold, (train_index, val_index) in enumerate(skf.split(X_text, Y), 1):
    print(f'Fold {fold}/{n_splits}')
    # Split the data into training and validation sets
    x_train_text, x_val_text = X_text[train_index], X_text[val_index]
    x_train_audio, x_val_audio = X_audio[train_index], X_audio[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    # Create a new model instance for each fold
    model = create_model()

    # Train the model
    history = model.fit([x_train_text, x_train_audio], y_train, batch_size=16, epochs=20, validation_data=([x_val_text, x_val_audio], y_val), callbacks=[lr_scheduler], verbose=1)

    # Evaluate the model
    scores = model.evaluate([x_val_text, x_val_audio], y_val, verbose=1)
    cv_scores.append(scores[1])  # Assuming scores[1] is the accuracy or relevant metric
    
    # Predict labels for validation set
    y_pred = model.predict([x_val_text, x_val_audio])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    # Print classification report for the current fold
    print(classification_report(y_true, y_pred_classes))

    # Plot loss and accuracy curves for the current fold
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Calculate the average performance across all folds
average_performance = np.mean(cv_scores)
print(f'Average Performance across all folds: {average_performance}')
