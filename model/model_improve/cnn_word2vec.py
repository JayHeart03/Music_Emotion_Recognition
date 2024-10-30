import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,f1_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical,pad_sequences
from keras.optimizers import Adam
from keras.regularizers import l2
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
from keras.callbacks import LearningRateScheduler

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Function to preprocess text NR+SR+LC
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in tokens if word.lower() not in stop_words]
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

# Calculate the total word count in the processed lyrics
word_count = sum(len(text.split()) for text in data['processed_lyrics'])
print(data[['lyrics', 'processed_lyrics']].head())
print(f"Total number of words in the processed lyrics: {word_count}")

# Initialize the LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['mood'])
labels = to_categorical(encoded_labels)

# Fit the tokenizer on the processed lyrics
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['processed_lyrics'])
sequences = tokenizer.texts_to_sequences(data['processed_lyrics'])

word_index = tokenizer.word_index
print(f'Found {len(word_index)} unique tokens.')

max_sequence_length = 250
print(f'Max sequence length: {max_sequence_length}')

data = pad_sequences(sequences, maxlen=max_sequence_length)
print(f'Shape of data tensor: {data.shape}')

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

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

# Define a learning rate scheduler
def exponential_decay(epoch):
    initial_lr = 0.0005  
    k = 0.1  
    lr = initial_lr * np.exp(-k * epoch)
    return lr

lr_scheduler = LearningRateScheduler(exponential_decay)

# Build the model
model = Sequential()
model.add(Embedding(vocab_size,
                    embedding_dim,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=max_sequence_length,
                        trainable=False))
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

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=20,
                    validation_data=(x_val, y_val),
                    callbacks=[lr_scheduler],
                    verbose=1)

# Evaluate the model
scores = model.evaluate(x_val, y_val, verbose=1)

for layer in model.layers:
    weights = layer.get_weights()
    print(f"Layer: {layer.name}")
    for i, weight in enumerate(weights):
        print(f"Weight {i}: shape {weight.shape}")

print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])

y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))
print(f1_score(y_true, y_pred_classes, average="weighted"))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Save the model and tokenizer
# model.save('model_improve/cnn_bow37.h5')

# import pickle
# with open('model_improve/tokenizer.pkl2', 'wb') as f:
#     pickle.dump(tokenizer, f)

# word2vec_model.save('model_improve/word2vec_model2')

# with open('model_improve/label_encoder2.pkl', 'wb') as f:
#     pickle.dump(label_encoder, f)

