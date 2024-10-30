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

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define the text preprocessing function SR+LC+NR
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)


# Load and preprocess the dataset
data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
#shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)

# Define continuous features
happy_songs = data[data['mood'] == 'happy']
 
if len(happy_songs) > 150:
    np.random.seed(42)
    drop_indices = np.random.choice(happy_songs.index, size=90, replace=False)
    data = data.drop(drop_indices)
    print('Downsampling successful!')
    print(f'Number of songs in each mood after downsampling:\n{data["mood"].value_counts()}')

# Define continuous features
continuous_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'loudness']
# categorical_features = ['key', 'mode', 'time_signature']

# Data preprocessing for continuous features
continuous_transformer = Pipeline(steps=[('scaler',  StandardScaler())])
# categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing for continuous and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_features),
        # ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the continuous features
audio_features = data[continuous_features]
processed_audio_features = preprocessor.fit_transform(audio_features)

# Tokenize and pad the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['processed_lyrics'])
sequences = tokenizer.texts_to_sequences(data['processed_lyrics'])

# Set the maximum sequence length to 250
max_sequence_length = 250
data_padded = pad_sequences(sequences, maxlen=max_sequence_length)

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['mood'])
labels = to_categorical(encoded_labels)

# Split the data into training and validation sets
x_train_text, x_val_text, y_train, y_val = train_test_split(data_padded, labels, test_size=0.2, random_state=42, stratify=labels)
print(x_train_text.shape, x_val_text.shape)
print(y_train.shape, y_val.shape)
x_train_audio, x_val_audio = train_test_split(processed_audio_features, test_size=0.2, random_state=42, stratify=labels)
print(x_train_audio.shape, x_val_audio.shape)

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

# Model building
text_input = Input(shape=(max_sequence_length,), dtype='int32', name='Text_Input')
embedding_layer = Embedding(vocab_size, embedding_dim, embeddings_initializer=Constant(embedding_matrix), 
                            input_length=max_sequence_length, trainable=False, name='Embedding')(text_input)
# Text
conv1 = Conv1D(128, 5, activation='relu', name='Conv_Text1')(embedding_layer)
maxpool1 = MaxPooling1D(5, name='Maxpool_text1')(conv1)
dropout1 = Dropout(0.2, name='Dropout_text1')(maxpool1)
conv2 = Conv1D(64, 5, activation='relu', name='Conv_Text2')(dropout1)
maxpool2 = MaxPooling1D(5, name='Maxpool_Text2')(conv2)
dropout2 = Dropout(0.2, name='Dropout_text2')(maxpool2)
conv3 = Conv1D(32, 5, activation='relu', name='Conv_text3')(dropout2)
dropout3 = Dropout(0.2, name='Dropout_text3')(conv3)
global_maxpool = GlobalMaxPooling1D(name='Global_maxpool')(dropout3)

# Audio
audio_input = Input(shape=(audio_features.shape[1],), name='Audio_Input')
dense_audio1 = Dense(64, activation='relu', name='Dense_Audio1')(audio_input)
dropout_audio1 = Dropout(0.2, name='Dropout_Audio1')(dense_audio1)
dense_audio2 = Dense(32, activation='relu', name='Dense_Audio2')(dropout_audio1)
dropout_audio2 = Dropout(0.2, name='Dropout_Audio2')(dense_audio2)
dense_audio3 = Dense(16, activation='relu', name='Dense_Audio3')(dropout_audio2)

# Merge the text and audio input
merged = concatenate([global_maxpool, dense_audio3], name='Merge')
classifier = Dense(len(labels[0]), activation='softmax', name='Classifier')(merged)

model = Model(inputs=[text_input, audio_input], outputs=classifier) 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define a learning rate scheduler
def exponential_decay(epoch):
    initial_lr = 0.0005
    k = 0.1
    lr = initial_lr * np.exp(-k * epoch)
    return lr

lr_scheduler = LearningRateScheduler(exponential_decay)

# Train the model
history = model.fit([x_train_text, x_train_audio], y_train, batch_size=16, epochs=20, validation_data=([x_val_text, x_val_audio], y_val), callbacks=[lr_scheduler], verbose=1)
print('Training completed!')

# save the model plot
# from keras.utils import plot_model
# plot_model(model, to_file='model/model_combine/cnn_denseNet_word2vec.png', show_shapes=True)

# print('Training history:', history.history)
# print('prameters:', model.count_params())
for layer in model.layers:
    weights = layer.get_weights()
    print(f"Layer: {layer.name}")
    for i, weight in enumerate(weights):
        print(f"Weight {i}: shape {weight.shape}")

# Evaluate the model
scores = model.evaluate([x_val_text, x_val_audio], y_val, verbose=1)
y_pred = model.predict([x_val_text, x_val_audio])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print(classification_report(y_true, y_pred_classes))
print(f'F1 Score: {f1_score(y_true, y_pred_classes, average="weighted"):.2f}')


plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()


plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()


# # Save the model
# model.save('text_audio_model.h5')
# # Save the label encoder
# import joblib
# joblib.dump(label_encoder, 'label_encoder.pkl')
# # Save the tokenizer
# import pickle
# with open('tokenizer.pkl', 'wb') as file:
#     pickle.dump(tokenizer, file)