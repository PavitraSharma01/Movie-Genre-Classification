import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess text data:
    - Convert to lowercase
    - Remove special characters
    - Remove extra spaces
    - Remove stopwords
    """
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

"""def load_and_preprocess_data():
    
    Load and preprocess the dataset.
    
    # Load training data
    train_df = pd.read_csv('data/train_data.txt', sep=' ::: ', header=None, engine='python', names=["id", "title", "genre", "description"])

    # Load test data
    test_df = pd.read_csv('data/test_data.txt', sep=' ::: ', header=None, engine='python', names=["id", "title", "description"])

    # Load test data solution (true labels)
    solution_df = pd.read_csv('data/test_data_solution.txt', sep=' ::: ', header=None, engine='python', names=["id", "title", "genre", "description"])

    # Convert 'id' columns to the same data type (e.g., string) and clean them
    test_df['id'] = test_df['id'].astype(str).str.strip()  # Remove extra spaces
    solution_df['id'] = solution_df['id'].astype(str).str.strip()  # Remove extra spaces

    # Debugging: Print the first few rows of 'id' columns
    print("First few 'id' values in test_df:")
    print(test_df['id'].head())

    print("First few 'id' values in solution_df:")
    print(solution_df['id'].head())

    # Merge test data with solutions for evaluation
    test_df = test_df.merge(solution_df[["id", "genre"]], on="id", how="left")

    # Debugging: Print the number of rows before and after merge
    print(f"Number of rows in test_df before merge: {len(test_df)}")
    print(f"Number of rows in solution_df: {len(solution_df)}")
    print(f"Number of rows in test_df after merge: {len(test_df)}")

    # Check for NaN values in the 'genre' column after merge
    nan_count = test_df['genre'].isna().sum()
    print(f"Number of NaN values in 'genre' column after merge: {nan_count}")

    # Drop rows with NaN values in the 'genre' column (if any)
    test_df = test_df.dropna(subset=['genre'])

    # Debugging: Print the number of rows after dropping NaN values
    print(f"Number of rows in test_df after dropping NaN values: {len(test_df)}")

    # Preprocess text data
    train_df['description_processed'] = train_df['description'].apply(preprocess_text)
    test_df['description_processed'] = test_df['description'].apply(preprocess_text)

    # Save preprocessed data
    train_df.to_csv('data/train_processed.csv', index=False)
    test_df.to_csv('data/test_processed.csv', index=False)
    return train_df, test_df"""

def load_and_preprocess_data():
    """
    Load and preprocess the dataset.
    """
    # Define the absolute path to the data folder
    data_dir = os.path.abspath('data')

    # Load training data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.txt'), sep=' ::: ', header=None, engine='python', names=["id", "title", "genre", "description"])

    # Load test data
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.txt'), sep=' ::: ', header=None, engine='python', names=["id", "title", "description"])

    # Load test data solution (true labels)
    solution_df = pd.read_csv(os.path.join(data_dir, 'test_data_solution.txt'), sep=' ::: ', header=None, engine='python', names=["id", "title", "genre", "description"])

    # Convert 'id' columns to the same data type (e.g., string) and clean them
    test_df['id'] = test_df['id'].astype(str).str.strip()
    solution_df['id'] = solution_df['id'].astype(str).str.strip()

    # Merge test data with solutions for evaluation
    test_df = test_df.merge(solution_df[["id", "genre"]], on="id", how="left")

    # Preprocess text data
    train_df['description_processed'] = train_df['description'].apply(preprocess_text)
    test_df['description_processed'] = test_df['description'].apply(preprocess_text)

    # Save preprocessed data
    train_df.to_csv(os.path.join(data_dir, 'train_processed.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test_processed.csv'), index=False)

    print("Data preprocessing complete. Preprocessed data saved to 'data/train_processed.csv' and 'data/test_processed.csv'.")
    return train_df, test_df

if __name__ == "__main__":
    load_and_preprocess_data()