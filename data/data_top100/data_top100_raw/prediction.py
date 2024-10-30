import os
import pandas as pd
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.utils import pad_sequences


def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

# Load the model and tokenizer
model = load_model('model/model_save/Top100prediction_model/cnn_densenet_word2vec.h5')

with open('model/model_save/Top100prediction_model/tokenizer_final_Q2.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the label encoder
with open('model/model_save/Top100prediction_model/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

max_sequence_length = 250

# Define the audio features
audio_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'loudness']

for year in range(2013, 2024):
    file_path = f'data/data_top100/data_top100_raw/{year}.csv'
    if not os.path.exists(file_path):
        print(f"Data file for year {year} not found.")
        continue

    data = pd.read_csv(file_path)

    # Preprocess the text data
    data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)
    sequences = tokenizer.texts_to_sequences(data['processed_lyrics'])
    data_padded = pad_sequences(sequences, maxlen=max_sequence_length)

    # Preprocess the audio features
    scaler = StandardScaler()
    audio_data = data[audio_features]
    processed_audio_features = scaler.fit_transform(audio_data)

    # Make predictions
    predictions = model.predict([data_padded, processed_audio_features])
    predicted_classes = np.argmax(predictions, axis=1)

    # Decode the predicted classes
    predicted_labels = label_encoder.inverse_transform(predicted_classes)

    data['predicted_mood'] = predicted_labels
    data.drop('processed_lyrics', axis=1, inplace=True)

    output_file_path = f'data/data_top100/data_top100_predicted/{year}_predictions.csv'
    data.to_csv(output_file_path, index=False)

    print(f"Predictions for {year} saved to {output_file_path}")
