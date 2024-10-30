import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import seaborn as sns
import matplotlib.pyplot as plt

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # Apply lemmatization
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    # Define English stopwords
    stop_words = set(stopwords.words('english'))
    # Remove stopwords and non-alphabetic tokens
    cleaned = [word for word in lemmatized if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

data = pd.read_csv('data/MoodyLyrics.csv')
# data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocess the lyrics
data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)  
# Calculate the total word count in the processed lyrics
word_count = sum(len(text.split()) for text in data['processed_lyrics'])
print(data[['lyrics', 'processed_lyrics']].head())
print(f"Total number of words in the processed lyrics: {word_count}")

# Initialize the tokenizer
tokenizer = Tokenizer() 
# Fit the tokenizer on the processed lyrics
tokenizer.fit_on_texts(data['processed_lyrics'])
# Convert lyrics to sequences of integers
sequences = tokenizer.texts_to_sequences(data['processed_lyrics'])
# Calculate the length of each lyrics sequence
lengths = [len(x) for x in sequences]
# Determine the 97th percentile of lengths
print(np.percentile(lengths, 97))

# Plot the distribution of lyrics lengths
plt.figure(figsize=(10, 6))
sns.histplot(lengths, kde=True)
plt.title('Distribution of Lyrics Length')
plt.xlabel('Number of Words')
plt.ylabel('Number of Lyrics')
plt.show()
