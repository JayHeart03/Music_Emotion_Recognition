import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input, concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import gensim
import matplotlib.pyplot as plt
import pickle

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define the text preprocessing function Lemma+NR+SR
def preprocess_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    stop_words = set(stopwords.words('english'))
    cleaned = [word for word in lemmatized if word not in stop_words and word.isalpha()]
    return ' '.join(cleaned)


data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
#shuffle the data
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
# data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Define continuous features
continuous_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
# categorical_features = ['key', 'mode', 'time_signature']

# Data preprocessing for continuous features
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), continuous_features),
    # ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Preprocess the audio features
audio_features = preprocessor.fit_transform(data)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['processed_lyrics'])
sequences = tokenizer.texts_to_sequences(data['processed_lyrics'])
text_data = pad_sequences(sequences, maxlen=250) # set the maximum sequence length to 250

# One-hot encode the target variable
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data['mood'])
y = to_categorical(integer_encoded)

# Split the data into training and testing sets
X_train_text, X_test_text, X_train_audio, X_test_audio, y_train, y_test = train_test_split(text_data, audio_features, y, test_size=0.2, random_state=42, stratify=y)

# Load the pre-trained Word2Vec model
word2vec_path = 'data/data_moody/GoogleNews-vectors-negative300.bin'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# Create an embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 300
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in word2vec_model.key_to_index:
        embedding_vector = word2vec_model[word]
        embedding_matrix[i] = embedding_vector

# Build the combined model
def build_combined_model(input_dim_text, embedding_matrix, input_length_text, audio_input_shape, output_dim, embedding_dim=300):
    text_input_layer = Input(shape=(input_length_text,))
    text_layer = Embedding(input_dim=input_dim_text, output_dim=embedding_dim, weights=[embedding_matrix], input_length=input_length_text, trainable=False)(text_input_layer)
    text_layer = Dropout(0.2)(text_layer)
    text_layer = Bidirectional(LSTM(100))(text_layer)

    audio_input_layer = Input(shape=(audio_input_shape,))
    audio_layer = Dense(64, activation='relu')(audio_input_layer)
    audio_layer = Dropout(0.2)(audio_layer)
    audio_layer = Dense(32, activation='relu')(audio_layer)
    audio_layer = Dropout(0.2)(audio_layer)
    audio_layer = Dense(16, activation='relu')(audio_layer)

    merged_layer = concatenate([text_layer, audio_layer])
    output_layer = Dense(output_dim, activation='softmax')(merged_layer)

    combined_model = Model(inputs=[text_input_layer, audio_input_layer], outputs=output_layer)
    combined_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return combined_model 

# Instantiate the combined model
combined_model = build_combined_model(vocab_size, embedding_matrix, 250, X_train_audio.shape[1], y.shape[1])

# Train the model
combined_model.fit([X_train_text, X_train_audio], y_train, epochs=30, batch_size=16, validation_split=0.1)

plt.plot(combined_model.history.history['accuracy'])
plt.plot(combined_model.history.history['val_accuracy'])
plt.title('Combined Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(combined_model.history.history['loss'])
plt.plot(combined_model.history.history['val_loss'])
plt.title('Combined Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model
y_pred = combined_model.predict([X_test_text, X_test_audio])
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print("Combined Model Performance:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Save the model
# combined_model.save('model_text_audio/bilstm_word_audio.h5')

# Save the tokenizer
# with open('model_text_audio/tokenizer_bilstm.pkl', 'wb') as file:
#     pickle.dump(tokenizer, file)
