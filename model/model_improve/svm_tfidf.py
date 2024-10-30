import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Function to preprocess text NR+SR+LC
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(cleaned)

data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
#shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data['processed_lyrics'] = data['lyrics'].apply(preprocess_text)

# Downsample the 'happy' mood class
happy_songs = data[data['mood'] == 'happy']

if len(happy_songs) > 150:
    np.random.seed(42)
    drop_indices = np.random.choice(happy_songs.index, size=90, replace=False)
    data = data.drop(drop_indices)
    print('Downsampling successful!')
    print(f'Number of songs in each mood after downsampling:\n{data["mood"].value_counts()}')

word_count = sum(len(text.split()) for text in data['processed_lyrics'])
print(data[['lyrics', 'processed_lyrics']].head())

print(f"Total number of words in the processed lyrics: {word_count}")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['mood'])

# Encode the target variable
mood_classes = label_encoder.classes_
encoded_values = range(len(mood_classes))
mood_mapping = dict(zip(mood_classes, encoded_values))
print(mood_mapping)

x = data['processed_lyrics']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=0.001, ngram_range=(1, 1))

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
print("TFIDF Vectorizer Shape:", X_train_tfidf.shape)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("TFIDF Vectorizer Shape:", X_test_tfidf.shape)

print("TFIDF Vectorizer Vocabulary Size:", y_train.shape)
print("TFIDF Vectorizer Vocabulary Size:", y_test.shape)

# Print the first 10 words in the vocabulary
for i, (word, index) in enumerate(tfidf_vectorizer.vocabulary_.items()):
    if i < 10:
        print(f"Word: '{word}', Index: {index}")
print("TFIDF Vectorizer Shape:", X_train_tfidf)
print("TFIDF Vectorizer Shape:", X_test_tfidf)

# Create a svm c=0.36
svc = SVC(kernel='linear', C=0.36)
print("SVM Parameters:\n", svc.get_params())

svc.fit(X_train_tfidf, y_train)

# Predict the labels
y_pred = svc.predict(X_test_tfidf)
# Evaluate the model
classification_report_result = classification_report(y_test, y_pred)
accuracy_score_result = accuracy_score(y_test, y_pred)
print("Classification Report:\n", classification_report_result)
print("f1_score:", f1_score(y_test, y_pred, average='weighted'))
print("Accuracy:", accuracy_score_result)
print("confusion_matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
# import joblib
# joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

# import joblib
# joblib.dump(svc, 'svm_tfidf.joblib')

# learning curve
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")
    
    plt.legend(loc="best")
    plt.show()
    print("Training Score:", train_scores_mean[-1]) 
    print("Test Score:", test_scores_mean[-1])
    print("Cross-Validation Score:", test_scores_mean[-1])


plot_learning_curve(svc,X_train_tfidf, y_train)



# def plot_top_tfidf_features(vectorizer, tfidf_result, top_n=20):
#     feature_names = vectorizer.get_feature_names_out()
#     sorted_items = sorted(zip(feature_names, tfidf_result.sum(axis=0).tolist()[0]), key=lambda x: -x[1])[:top_n]
    
#     scores = [item[1] for item in sorted_items]
#     terms = [item[0] for item in sorted_items]
    
#     fig, ax = plt.subplots()
#     ax.barh(terms, scores)
#     ax.set_xlabel('TFIDF Score')
#     ax.set_ylabel('Terms')
#     ax.set_title('Top TFIDF Features')
#     plt.gca().invert_yaxis()
#     plt.show()

# plot_top_tfidf_features(tfidf_vectorizer, X_train_tfidf)

# from wordcloud import WordCloud

# def generate_wordcloud(tfidf_result, vectorizer):
#     scores = zip(vectorizer.get_feature_names_out(), np.asarray(tfidf_result.sum(axis=0)).ravel())
#     sorted_scores = dict(sorted(scores, key=lambda x: x[1], reverse=True))
    
#     wordcloud = WordCloud(background_color='white', width=800, height=600)
#     wordcloud.generate_from_frequencies(sorted_scores)
    
#     plt.figure(figsize=(10, 8))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.show()

# generate_wordcloud(X_train_tfidf, tfidf_vectorizer)

# def plot_multiclass_feature_importance(classifier, vectorizer, label_encoder, top_n=10):
#     feature_names = vectorizer.get_feature_names_out()
#     svm_coefficients = classifier.coef_.toarray()

#     for i, class_label in enumerate(label_encoder.classes_):
#         class_coefs = svm_coefficients[i]
#         print(f"Class Label: {class_label}")
#         print(f"Class Index: {i}")
        
#         top_positive_features = sorted(zip(feature_names, class_coefs), key=lambda x: x[1], reverse=True)[:top_n]
#         top_negative_features = sorted(zip(feature_names, class_coefs), key=lambda x: x[1])[:top_n]

#         top_features = top_positive_features + top_negative_features
#         terms = [item[0] for item in top_features]
#         coefficients = [item[1] for item in top_features]

#         plt.figure(figsize=(10, 10))
#         plt.barh(terms, coefficients)
#         plt.xlabel('SVM Coefficient Value')
#         plt.ylabel('Terms')
#         plt.title(f'Top Positive and Negative Features for Class {class_label}')
#         plt.gca().invert_yaxis()
#         plt.show()

# plot_multiclass_feature_importance(svc, tfidf_vectorizer, label_encoder)


# from sklearn.decomposition import PCA
# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.decomposition import PCA
# from matplotlib.colors import ListedColormap

# pca = PCA(n_components=2)
# X_train_pca = pca.fit_transform(X_train_tfidf.toarray())


# plt.figure(figsize=(16, 10))

# colors = ['r', 'g', 'b', 'y'] 
# markers = ['o', 's', 'D', 'v']
# for i, class_label in enumerate(label_encoder.classes_):
#     plt.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1], c=colors[i], marker=markers[i], label=class_label)

# h = 0.02
# x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
# y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Z = svc.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
# Z = Z.reshape(xx.shape)

# cmap_background = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA'])
# cmap_foreground = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])
# plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.3)
# plt.colorbar()
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())

# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('Decision Boundary and Scatter Plot')
# plt.legend(loc='best')
# plt.show()


