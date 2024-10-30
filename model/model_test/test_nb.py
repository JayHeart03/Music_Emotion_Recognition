
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score,f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Define the text preprocessing functions Lemma+SR+LC+NR
def preprocess_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in lemmatized if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

# continuous_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'loudness']
# # categorical_features = ['key', 'mode', 'time_signature']
# continuous_transformer = Pipeline(steps=[('scaler', StandardScaler())])
# # categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
# preprocessor = ColumnTransformer(transformers=[('num', continuous_transformer, continuous_features)])


test_data = pd.read_csv('data/data_moody/MoodyLyrics4Q.csv')

test_data['processed_lyrics'] = test_data['lyrics'].apply(preprocess_text)

# Load the text transformer and the model
text_transformer = joblib.load('model/model_save/tokenizer/nb_countvectorizer.joblib')

X_test_text = text_transformer.transform(test_data['processed_lyrics'])

test_data['tokens'] = test_data['lyrics'].apply(preprocess_text)

nb_text = joblib.load('model/model_save/lyrcis_only/nb_bow.joblib')

y_pred_test = nb_text.predict(X_test_text)#lyrcis and audio features


print("Test Performance with Combined Features:")
print(classification_report(test_data['mood'], y_pred_test))
print(f"Accuracy: {accuracy_score(test_data['mood'], y_pred_test)}")
print(f"F1 Score: {f1_score(test_data['mood'], y_pred_test, average='weighted')}")

