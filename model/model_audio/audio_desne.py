import pandas as pd
import numpy as np

# Importing libraries for deep learning and data processing
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Reading the dataset
data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
# Randomly shuffling the data and resetting the index
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Selecting songs with the 'happy' mood
happy_songs = data[data['mood'] == 'happy']

# Downsampling if there are more than 150 'happy' songs
if len(happy_songs) > 150:
    np.random.seed(42)
    drop_indices = np.random.choice(happy_songs.index, size=90, replace=False)
    data = data.drop(drop_indices)
    print('Downsampling successful!')
    print(f'Number of songs in each mood after downsampling:\n{data["mood"].value_counts()}')

# Selecting continuous features
continuous_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'loudness']
# Initializing a StandardScaler for feature scaling
preprocessor = StandardScaler()
processed_audio_features = preprocessor.fit_transform(data[continuous_features])

# Encoding the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['mood'])
# Converting labels to one-hot encoding
labels = to_categorical(encoded_labels)

# Splitting the data into training and validation sets
x_train_audio, x_val_audio, y_train, y_val = train_test_split(processed_audio_features, labels, test_size=0.2, random_state=42, stratify=labels)

# Defining the neural network structure
audio_input = Input(shape=(x_train_audio.shape[1],), name='audio_input')
dense_audio = Dense(64, activation='relu')(audio_input)
dropout_audio = Dropout(0.2)(dense_audio)
dense_audio = Dense(32, activation='relu')(dense_audio)
dropout_audio = Dropout(0.2)(dense_audio)
dense_audio = Dense(16, activation='relu')(dense_audio)
classifier = Dense(len(labels[0]), activation='softmax')(dense_audio)

# Building and compiling the model
audio_model = Model(inputs=audio_input, outputs=classifier)
audio_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Defining an exponential decay function for the learning rate
def exponential_decay(epoch):
    initial_lr = 0.001
    k = 0.1
    lr = initial_lr * np.exp(-k * epoch)
    return lr

lr_scheduler = LearningRateScheduler(exponential_decay)

# Training the model
history = audio_model.fit(x_train_audio, y_train, epochs=30, batch_size=16, validation_data=(x_val_audio, y_val), callbacks=[lr_scheduler])
# Evaluating the model
scores = audio_model.evaluate(x_val_audio, y_val, verbose=1)

# Making predictions and printing metrics
y_pred = audio_model.predict(x_val_audio)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

print(classification_report(y_true, y_pred_classes))
print(f'F1 Score: {f1_score(y_true, y_pred_classes, average="weighted"):.2f}')
