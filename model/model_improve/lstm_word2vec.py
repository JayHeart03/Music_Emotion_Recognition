import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical,pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, InputLayer, Conv1D
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in lemmatized if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
#shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

happy_songs = data[data['mood'] == 'happy']

if len(happy_songs) > 150:
    np.random.seed(42)
    drop_indices = np.random.choice(happy_songs.index, size=90, replace=False)
    data = data.drop(drop_indices)
    print('Downsampling successful!')
    print(f'Number of songs in each mood after downsampling:\n{data["mood"].value_counts()}')

data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)  
word_count = sum(len(text.split()) for text in data['processed_lyrics'])
print(data[['lyrics', 'processed_lyrics']].head())
print(f"Total number of words in the processed lyrics: {word_count}")

# Initialize the tokenizer
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(data['processed_lyrics'])
sequences = tokenizer.texts_to_sequences(data['processed_lyrics'])
# len_seq =  [len(x) for x in sequences]
# print(np.percentile(len_seq,95))

# Pad the sequences so each sequence is the same length
data_pad = pad_sequences(sequences, maxlen=250)

# One-hot encode the target variable
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data['mood'])
y = to_categorical(integer_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_pad, y, test_size=0.2, random_state=42, stratify=y)

from gensim.models import KeyedVectors

#Load the pre-trained Word2Vec model
word2vec_path = 'data/data_moody/GoogleNews-vectors-negative300.bin'
word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# Create an embedding matrix
embedding_dim = 300  
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Fill the embedding matrix with the word2vec embeddings
for word, i in tokenizer.word_index.items():
    if i >= vocab_size:
        continue
    try:
        embedding_vector = word2vec[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except KeyError:
        pass 

# Build the model
model = Sequential([
    InputLayer(input_shape=(250,)), 
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=250, trainable=False),
    Dropout(0.2), 
    LSTM(100), 
    Dense(4, activation='softmax',activity_regularizer=l2(0.001))
])
# Define a learning rate scheduler
learning_rate = 0.00007

optimizer = Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 50
history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, y_test)
for layer in model.layers:
    weights = layer.get_weights()
    print(f"Layer: {layer.name}")
    for i, weight in enumerate(weights):
        print(f"Weight {i}: shape {weight.shape}")

print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()
