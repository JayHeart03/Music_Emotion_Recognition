import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score,f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Define the text preprocessing functions SR+LC+NR
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

# Define the continuous and categorical features
continuous_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'loudness']
# categorical_features = ['key', 'mode', 'time_signature']
continuous_transformer = Pipeline(steps=[('scaler', StandardScaler())])
# categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', continuous_transformer, continuous_features)])

test_data = pd.read_csv('data/data_moody/MoodyLyrics4Q.csv') 

# Preprocess the lyrics
test_data['processed_lyrics'] = test_data['lyrics'].apply(preprocess_text)
text_transformer = joblib.load('model/model_save/tokenizer/svm_tfidf_vectorizer.joblib')

# Load the text transformer and the model
X_test_text = text_transformer.transform(test_data['processed_lyrics'])

audio_features = test_data[continuous_features]
processed_audio_features = preprocessor.fit_transform(audio_features)

# Combine the text and audio features
X_test_combined = np.hstack((X_test_text.toarray(), processed_audio_features))

svm_combined = joblib.load('model/model_save/combine/svm_tfidf_early.joblib')# combine the text and audio features
# svm_text = joblib.load('model/model_save/lyrcis_only/svm_tfidf.joblib')# only text features
ture = test_data['mood']

# Label encoding
label_encoder = LabelEncoder()
ture = label_encoder.fit_transform(ture)

# predicted_classes = svm_text.predict(X_test_text)   # only text features
predicted_classes = svm_combined.predict(X_test_combined)# combine the text and audio features
predicted_classes_num = label_encoder.transform(predicted_classes)

print("Classification Report:")
print(classification_report(ture, predicted_classes_num))
print("F1 Score:")
print(classification_report(ture, predicted_classes_num, output_dict=True)['weighted avg']['f1-score'])
