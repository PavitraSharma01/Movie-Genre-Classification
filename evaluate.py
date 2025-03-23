# evaluate.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

"""def evaluate_model():
    
    Evaluate the trained model on the test data.
    
    # Load preprocessed test data
    test_df = pd.read_csv('data/test_processed.csv')

    # Load the trained model and TF-IDF vectorizer
    model = joblib.load('models/genre_classifier.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')

    # Convert test data to numerical vectors
    X_test = tfidf.transform(test_df['description_processed']).toarray()
    y_test = test_df['genre']

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save predictions
    test_df['predicted_genre'] = y_pred
    test_df.to_csv('data/test_predictions.csv', index=False)
    print("Predictions saved to 'data/test_predictions.csv'.")"""
def evaluate_model():
    """
    Evaluate the trained model on the test data.
    """
    # Load preprocessed test data
    test_df = pd.read_csv('data/test_processed.csv')

    # Check for NaN values in the 'genre' column
    nan_count = test_df['genre'].isna().sum()
    print(f"Number of NaN values in 'genre' column: {nan_count}")

    # Drop rows with NaN values in the 'genre' column (if any)
    test_df = test_df.dropna(subset=['genre'])

    # Load the trained model and TF-IDF vectorizer
    model = joblib.load('models/genre_classifier.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')

    # Convert test data to numerical vectors
    X_test = tfidf.transform(test_df['description_processed']).toarray()
    y_test = test_df['genre']

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save predictions
    test_df['predicted_genre'] = y_pred
    test_df.to_csv('data/test_predictions.csv', index=False)
    print("Predictions saved to 'data/test_predictions.csv'.")


if __name__ == "__main__":
    evaluate_model()