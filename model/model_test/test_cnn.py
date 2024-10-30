import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from keras.models import load_model
from keras.utils import pad_sequences, to_categorical
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Define the text preprocessing function SR+LC+NR
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

data = pd.read_csv('data/data_moody/MoodyLyrics4Q.csv')

# Randomly shuffle the data and reset the index
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)
print(data['processed_lyrics'])

# Load the Tokenizer
with open('model/model_save/tokenizer/cnn_tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Pad the sequences
sequences = tokenizer.texts_to_sequences(data['processed_lyrics'])

word_index = tokenizer.word_index
unique_words_count = len(word_index)

# Print the number of unique words in the Tokenizer's vocabulary
print("Number of unique words:", unique_words_count)

word_index = tokenizer.word_index

test_words = set()
for text in data['processed_lyrics']:
    test_words.update(text.split())

unseen_words = test_words - set(word_index.keys())
unseen_words_count = len(unseen_words)

# Print the number of words in the test set not seen in the Tokenizer's vocabulary
print(f"Number of words in the test set not seen in the Tokenizer's vocabulary: {unseen_words_count}")

# Pad the sequences
max_sequence_length = 250
data_padded = pad_sequences(sequences, maxlen=max_sequence_length)

# Preprocess the audio features
continuous_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'loudness']
# categorical_features = ['key', 'mode', 'time_signature']
continuous_transformer = Pipeline(steps=[('scaler', StandardScaler())])
# categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', continuous_transformer, continuous_features)])
audio_features = data[continuous_features]
processed_audio_features = preprocessor.fit_transform(audio_features)

# print("Loading model...")
model = load_model('model/model_save/combine/cnn_densenet_word2vec.h5') # test lrycis and audio
# model_text = load_model('model/model_save/lyrcis_only/cnn_word2vec35.7.h5') # test lrycis only
model.summary()

# from keras.utils import plot_model
# plot_model(model, to_file='model_text_audio/cnn_final_stander2.png', show_shapes=True)

# embedding_layer = model.layers[5]
# weights = embedding_layer.get_weights()[0]
# print(weights.shape)

optimizer = model.optimizer
current_lr = optimizer.lr.numpy()
print(f"Current learning rate: {current_lr}")

# Load the label encoder
with open('model/model_save/tokenizer/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

encoded_labels = label_encoder.fit_transform(data['mood'])
categorical_labels = to_categorical(encoded_labels)
print(categorical_labels)

predictions = model.predict([data_padded, processed_audio_features]) # test lrycis and audio
# predictions = model_text.predict([data_padded]) # test lrycis only
predicted_classes = np.argmax(predictions, axis=1)

true_classes = label_encoder.transform(data['mood'])
print(true_classes)

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print("Classification Report:")
print(classification_report(true_classes, predicted_classes))
print("F1 Score: ", f1_score(true_classes, predicted_classes, average="weighted"))

