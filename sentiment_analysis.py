import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Helper function
def print_sample_and_class(data, class_=1, n_samples=5):
    index = data[data['y'] == class_]['X'].index[0:n_samples]
    for i in index:
        print(f"Sample {i} of class {class_}")
        print(data[data['y'] == class_]['X'][i])
        print("-" * 50)

# Load the training data
df = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX08RAEN/sentiment-text-threeclass/train.txt",
    header=None,
    sep="\\|\\|\\|",
    names=['y', 'X']
)

# View the first few rows
print("First 5 rows of the dataset:")
print(df.head())
print()

# Check the shape
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print()

# Check class distribution
print("Class distribution:")
print(df['y'].value_counts())

print("\n--- POSITIVE REVIEWS (class 1) ---")
print_sample_and_class(df, class_=1, n_samples=2)

print("\n--- NEGATIVE REVIEWS (class -1) ---")
print_sample_and_class(df, class_=-1, n_samples=2)

print("\n--- NEUTRAL REVIEWS (class 0) ---")
print_sample_and_class(df, class_=0, n_samples=2)

# ============================================
# RULE-BASED SENTIMENT ANALYSIS
# ============================================

# Define positive and negative words
good_words = ['love', 'good', 'excellent', 'great', 'charming']
bad_words = ['hate', 'bad', 'brutal', 'damnable', 'deplorable', 
             'detestable', 'disastrous', 'dreadful']

# Initialize score column
df["score"] = 0

# Subtract 1 for each bad word found
for bad_word in bad_words:
    df["score"] -= df["X"].str.casefold().str.count(bad_word)

# Add 1 for each good word found
for good_word in good_words:
    df["score"] += df["X"].str.casefold().str.count(good_word)

# View the results
print("Dataset with scores:")
print(df.head(10))

# Find the most positive review (highest score)
print("\n--- MOST POSITIVE REVIEW (highest score) ---")
print(df.loc[df["score"].idxmax(), 'X'])
print(f"Score: {df['score'].max()}")

# Find the most negative review (lowest score)
print("\n--- MOST NEGATIVE REVIEW (lowest score) ---")
print(df.loc[df["score"].idxmin(), 'X'])
print(f"Score: {df['score'].min()}")

# See average score by class
print("\n--- AVERAGE SCORE BY CLASS ---")
print(df.groupby('y')['score'].mean())

# ============================================
# EVALUATE RULE-BASED APPROACH
# ============================================

# Assign predictions based on score
df['yhat'] = 0  # Default to neutral
df.loc[df['score'] > 0, 'yhat'] = 1    # Positive if score > 0
df.loc[df['score'] < 0, 'yhat'] = -1   # Negative if score < 0

# Calculate accuracy
accuracy = np.mean(df['yhat'] == df['y'])
print(f"\n--- RULE-BASED ACCURACY ---")
print(f"Accuracy: {accuracy:.2%}")

# See the breakdown
print("\n--- PREDICTION DISTRIBUTION ---")
print(df['yhat'].value_counts())

print("\n--- ACTUAL VS PREDICTED ---")
print(pd.crosstab(df['y'], df['yhat'], rownames=['Actual'], colnames=['Predicted']))

# ============================================
# MACHINE LEARNING APPROACH
# ============================================

# Load training, validation, and test datasets
train_dataset = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX08RAEN/sentiment-text-threeclass/train.txt",
    header=None, sep="\\|\\|\\|", names=['y', 'X']
)

validation_dataset = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX08RAEN/sentiment-text-threeclass/dev.txt",
    header=None, sep="\\|\\|\\|", names=['y', 'X']
)

test_dataset = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX08RAEN/sentiment-text-threeclass/test.txt",
    header=None, sep="\\|\\|\\|", names=['y', 'X']
)

print("--- DATASET SIZES ---")
print(f"Training:   {train_dataset.shape[0]} samples")
print(f"Validation: {validation_dataset.shape[0]} samples")
print(f"Test:       {test_dataset.shape[0]} samples")

# ============================================
# BAG OF WORDS - Convert text to numbers
# ============================================
from sklearn.feature_extraction.text import CountVectorizer

# Create the vectorizer
vectorizer = CountVectorizer()

# Fit on training data and transform
X_train = vectorizer.fit_transform(train_dataset['X'])
y_train = train_dataset['y']

# Transform validation and test (don't fit again!)
X_val = vectorizer.transform(validation_dataset['X'])
y_val = validation_dataset['y']

X_test = vectorizer.transform(test_dataset['X'])
y_test = test_dataset['y']

print(f"\n--- BAG OF WORDS RESULT ---")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())} unique words")
print(f"X_train shape: {X_train.shape}")

# ============================================
# TRAIN LOGISTIC REGRESSION
# ============================================
from sklearn.linear_model import LogisticRegression

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Check accuracy on all datasets
train_accuracy = model.score(X_train, y_train)
val_accuracy = model.score(X_val, y_val)
test_accuracy = model.score(X_test, y_test)

print(f"\n--- MODEL ACCURACY ---")
print(f"Training accuracy:   {train_accuracy:.2%}")
print(f"Validation accuracy: {val_accuracy:.2%}")
print(f"Test accuracy:       {test_accuracy:.2%}")

# ============================================
# IMPROVE MODEL WITH GRIDSEARCH
# ============================================
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Create a pipeline: Vectorizer → Logistic Regression
pipe = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Define hyperparameters to try
param_grid = {
    "vectorizer__stop_words": [None, stopwords.words('english')],
    "vectorizer__min_df": [1, 5, 10],
    "vectorizer__max_df": [0.9, 1.0],
    "classifier__C": [0.1, 1, 10]
}

# Search for best parameters
print("\n--- SEARCHING FOR BEST PARAMETERS (this may take a minute)... ---")
grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(train_dataset['X'], train_dataset['y'])

# Results
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.2%}")

# Test with best model
best_model = grid_search.best_estimator_
test_accuracy_tuned = best_model.score(test_dataset['X'], test_dataset['y'])
print(f"Test accuracy (tuned model): {test_accuracy_tuned:.2%}")

# ============================================
# TF-IDF APPROACH
# ============================================
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF pipeline
pipe_tfidf = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Same parameter grid
param_grid_tfidf = {
    "vectorizer__stop_words": [None, stopwords.words('english')],
    "vectorizer__min_df": [1, 5, 10],
    "vectorizer__max_df": [0.9, 1.0],
    "classifier__C": [0.1, 1, 10]
}

print("\n--- SEARCHING WITH TF-IDF (this may take a minute)... ---")
grid_search_tfidf = GridSearchCV(pipe_tfidf, param_grid_tfidf, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_tfidf.fit(train_dataset['X'], train_dataset['y'])

print(f"\nBest parameters: {grid_search_tfidf.best_params_}")
print(f"Best cross-validation accuracy: {grid_search_tfidf.best_score_:.2%}")

# Test accuracy
test_accuracy_tfidf = grid_search_tfidf.best_estimator_.score(test_dataset['X'], test_dataset['y'])
print(f"Test accuracy (TF-IDF model): {test_accuracy_tfidf:.2%}")

# ============================================
# EXERCISE 1: BUILD YOUR OWN COUNTVECTORIZER
# ============================================

# Exercise 1a: Create word_to_idx dictionary
def create_word_to_idx(text_column):
    """
    Takes a column of text and returns a dictionary
    where each unique word is mapped to a unique number.
    """
    word_to_idx = {}
    idx = 0
    
    for document in text_column:
        # Split document into words and lowercase
        words = document.casefold().split()
        for word in words:
            if word not in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
    
    return word_to_idx

# Apply the function
word_to_idx = create_word_to_idx(train_dataset['X'])

print("\n--- EXERCISE 1a: WORD TO INDEX ---")
print(f"Total unique words: {len(word_to_idx)}")
print("\nFirst 10 words in dictionary:")
for i, (word, idx) in enumerate(word_to_idx.items()):
    if i < 10:
        print(f"  '{word}' → {idx}")

# Exercise 1b: Create TDM function
from scipy.sparse import dok_matrix

def create_tdm(text_column, word_to_idx):
    """
    Creates a Term-Document Matrix.
    Rows = documents, Columns = words, Values = word counts
    """
    n_docs = len(text_column)
    n_words = len(word_to_idx)
    
    # Create sparse matrix (efficient for large data)
    tdm = dok_matrix((n_docs, n_words), dtype=int)
    
    for doc_idx, document in enumerate(text_column):
        words = document.casefold().split()
        for word in words:
            if word in word_to_idx:
                word_idx = word_to_idx[word]
                tdm[doc_idx, word_idx] += 1
    
    return tdm

# Apply the function
X_train_custom = create_tdm(train_dataset['X'], word_to_idx)

print("\n--- EXERCISE 1b: CUSTOM TDM ---")
print(f"TDM shape: {X_train_custom.shape}")
print(f"(rows = documents, columns = unique words)")

# ============================================
# EXERCISE 2: SIMPLE LOGISTIC REGRESSION
# ============================================

# Convert sparse matrix to format sklearn likes
X_train_custom_csr = X_train_custom.tocsr()

# Create validation TDM using same word_to_idx
X_val_custom = create_tdm(validation_dataset['X'], word_to_idx).tocsr()
X_test_custom = create_tdm(test_dataset['X'], word_to_idx).tocsr()

# Train simple logistic regression (default settings)
simple_model = LogisticRegression(max_iter=1000)
simple_model.fit(X_train_custom_csr, train_dataset['y'])

# Calculate scores
train_score = simple_model.score(X_train_custom_csr, train_dataset['y'])
val_score = simple_model.score(X_val_custom, validation_dataset['y'])
test_score = simple_model.score(X_test_custom, test_dataset['y'])

print("\n--- EXERCISE 2: CUSTOM VECTORIZER + LOGISTIC REGRESSION ---")
print(f"Training accuracy:   {train_score:.2%}")
print(f"Validation accuracy: {val_score:.2%}")
print(f"Test accuracy:       {test_score:.2%}")

# ============================================
# EXPORT DATA FOR DASHBOARD (FIXED)
# ============================================
import os

# Create exports folder
os.makedirs('exports', exist_ok=True)

# 1. Model Comparison Data - ensure numbers are proper floats
model_comparison = pd.DataFrame({
    'Model': ['Rule-Based', 'Basic ML (BoW)', 'Custom Vectorizer', 'Tuned ML (BoW)', 'TF-IDF'],
    'Test_Accuracy': [23.60, 63.39, 64.03, 65.11, 66.02]
})
model_comparison.to_csv('exports/model_comparison.csv', index=False)
print(" Saved: exports/model_comparison.csv")

# 2. Class Distribution
class_dist = pd.DataFrame({
    'Sentiment': ['Positive', 'Negative', 'Neutral'],
    'Count': [3610, 3310, 1624]
})
class_dist.to_csv('exports/class_distribution.csv', index=False)
print(" Saved: exports/class_distribution.csv")

# 3. Test Predictions
best_model = grid_search_tfidf.best_estimator_
test_predictions = best_model.predict(test_dataset['X'])

sentiment_map = {1: 'Positive', -1: 'Negative', 0: 'Neutral'}
test_results = pd.DataFrame({
    'Actual': [sentiment_map[x] for x in test_dataset['y']],
    'Predicted': [sentiment_map[x] for x in test_predictions],
    'Review': test_dataset['X'].values
})
test_results.to_csv('exports/test_predictions.csv', index=False)
print("✅ Saved: exports/test_predictions.csv")

# 4. Confusion Matrix (long format for Power BI)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_dataset['y'], test_predictions, labels=[-1, 0, 1])
confusion_data = []
labels = ['Negative', 'Neutral', 'Positive']
for i, actual in enumerate(labels):
    for j, predicted in enumerate(labels):
        confusion_data.append({
            'Actual': actual,
            'Predicted': predicted,
            'Count': int(cm[i][j])
        })
confusion_df = pd.DataFrame(confusion_data)
confusion_df.to_csv('exports/confusion_matrix.csv', index=False)
print(" Saved: exports/confusion_matrix.csv")

# 5. Top Words
vectorizer_tfidf = best_model.named_steps['vectorizer']
classifier = best_model.named_steps['classifier']
feature_names = vectorizer_tfidf.get_feature_names_out()

top_words_list = []
for class_idx, class_label in enumerate([-1, 0, 1]):
    top_indices = classifier.coef_[class_idx].argsort()[-15:][::-1]
    sentiment = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}[class_label]
    for rank, idx in enumerate(top_indices, 1):
        top_words_list.append({
            'Sentiment': sentiment,
            'Rank': rank,
            'Word': feature_names[idx],
            'Weight': round(float(classifier.coef_[class_idx][idx]), 4)
        })

top_words_df = pd.DataFrame(top_words_list)
top_words_df.to_csv('exports/top_words_by_sentiment.csv', index=False)
print(" Saved: exports/top_words_by_sentiment.csv")

print("\n All files exported to 'exports' folder!")