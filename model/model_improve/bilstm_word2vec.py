import pandas as pd
import numpy as np
import nltk
import pickle
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

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Function to preprocess text Lemma+NR+SR
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Initialize a lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Lemmatize each token
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    # Define a set of English stopwords
    stop_words = set(stopwords.words('english'))
    # Remove stopwords and non-alphabetical characters
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
word_count = sum(len(text.split()) for text in data['processed_lyrics'])
print(data[['lyrics', 'processed_lyrics']].head())
print(f"Total number of words in the processed lyrics: {word_count}")

tokenizer = Tokenizer() 
tokenizer.fit_on_texts(data['processed_lyrics'])
sequences = tokenizer.texts_to_sequences(data['processed_lyrics'])
data_pad = pad_sequences(sequences, maxlen=250)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data['mood'])
y = to_categorical(integer_encoded)

X_train, X_test, y_train, y_test = train_test_split(data_pad, y, test_size=0.2, random_state=42, stratify=y)

from gensim.models import KeyedVectors

word2vec_path = 'data/data_moody/GoogleNews-vectors-negative300.bin'
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

model = Sequential([
    InputLayer(input_shape=(250,)),
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=250, trainable=False),
    Dropout(0.2),  
    Bidirectional(LSTM(100)), 
    Dense(4, activation='softmax') 
])

learning_rate = 0.0001 

# def lr_decay(epoch, lr):
#     decay_rate = 0.8
#     if epoch > 0:
#         new_lr = lr * decay_rate
#     else:
#         new_lr = lr
#     return new_lr

# lr_scheduler = LearningRateScheduler(lr_decay, verbose=1)

optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=3)

epochs = 30
history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, y_test)
for layer in model.layers:
    weights = layer.get_weights()
    print(f"Layer: {layer.name}")
    for i, weight in enumerate(weights):
        print(f"Weight {i}: shape {weight.shape}")

print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

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

 
# model.save('model_improve/bilstmtext_word2vec.h5')

# with open('model_improve/tokenizer_bilstmtext2_LEMMA_NR_SR.pkl', 'wb') as file:
#     tokenizer = pickle.dump(tokenizer, file)

# word2vec.save('model_improve/word2vecbilstmtext_model')


# with open('model_improve/label_encoder.pkl', 'wb') as file: