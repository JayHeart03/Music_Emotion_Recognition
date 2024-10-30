import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def preprocess_data(data):
    label_encoder = LabelEncoder()
    
    # List of features to scale
    features_scaler_scaler = ['danceability', 'energy', 'speechiness', 'acousticness', 
                            'instrumentalness', 'valence', 'liveness','loudness', 'tempo', 'duration_ms']

    # Setting up a pipeline with StandardScaler for feature scaling
    standard_transformer = Pipeline(steps=[
        ('scaler',  StandardScaler())
    ])

    # Preprocessor that applies the standard scaling to the specified features
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', standard_transformer, features_scaler_scaler),
            # Placeholder for potential categorical transformation
        ])

    # Transforming the features and encoding the target variable
    X = preprocessor.fit_transform(data)
    y = label_encoder.fit_transform(data['mood'])
    return X, y

# Function to evaluate different machine learning models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model.__class__.__name__}")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("-" * 80)

data = pd.read_csv('data/data_moody/MoodyLyrics.csv')
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X, y = preprocess_data(data)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Defining the models to be evaluated
models = [
    LogisticRegression(),
    SVC(),
    RandomForestClassifier(),
    XGBClassifier(),
    # Other models can be added here
]

for model in models:
    evaluate_model(model, X_train, X_test, y_train, y_test)
