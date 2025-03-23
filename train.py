import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_model():
    """
    Train a machine learning model on the preprocessed data.
    """
    # Load preprocessed training data
    train_df = pd.read_csv('data/train_processed.csv')

    # Convert text to numerical vectors using TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, max_df=0.8, min_df=5)
    X_train = tfidf.fit_transform(train_df['description_processed']).toarray()
    y_train = train_df['genre']

    # Split the data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Define the model with class weights
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    # model = RandomForestClassifier(n_estimators=100, class_weight='balanced')

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'C': [0.1, 1.0, 10.0],  # For Logistic Regression
        # 'n_estimators': [100, 200],  # For Random Forest
        # 'max_depth': [10, 20],  # For Random Forest
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_split, y_train_split)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the validation set
    y_val_pred = best_model.predict(X_val)
    print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))

    # Save the trained model and TF-IDF vectorizer
    joblib.dump(best_model, 'models/genre_classifier.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

    print("Model training complete. Model saved to 'models/genre_classifier.pkl'.")

if __name__ == "__main__":
    train_model()