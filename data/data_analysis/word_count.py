import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Function to preprocess and tokenize text
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

df = pd.read_csv('data/cleaned3_lyrics_and_features_data_copy.csv')

# Preprocess text data
df['processed_text'] = df['lyrics'].apply(preprocess_text)

# Initialize a dictionary to hold word counts for each category
category_word_counts = {category: Counter() for category in df['mood'].unique()}

# Count words in each category
for _, row in df.iterrows():
    words = row['processed_text'].split()  # Split the processed text into words
    category_word_counts[row['mood']].update(words)

# Get top 5 words for each category
top_words_per_category = {category: counts.most_common(5) for category, counts in category_word_counts.items()}

for category, top_words in top_words_per_category.items():
    print(f"Top words for {category}:")
    for word, count in top_words:
        print(f"  {word}: {count}")
    print()
