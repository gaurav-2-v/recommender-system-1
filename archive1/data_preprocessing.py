import os
import subprocess

# Check if the pickle file exists. If not, run the preprocessing script.
if not os.path.exists("model.pkl"):
    subprocess.run(["python", "data_preprocessing.py"])
import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

# --- STEP 1: LOAD THE DATA ---
# Ensure you have downloaded the TMDB 5000 dataset from Kaggle and placed 
# the two CSV files in the same folder as this script.
if not os.path.exists('tmdb_5000_movies.csv') or not os.path.exists('tmdb_5000_credits.csv'):
    print("ERROR: Dataset not found!")
    print("Please download 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' from Kaggle and place them in this folder.")
    exit()

print("Loading datasets...")
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# --- STEP 2: MERGE AND CLEAN DATA ---
print("Merging and cleaning data...")
movies = movies.merge(credits, on='title')
# Keep only necessary columns for a content-based recommender
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# Helper functions to extract text from stringified lists/dictionaries
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Apply the helper functions to extract data
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces between words (e.g., 'Science Fiction' -> 'ScienceFiction') 
# so the model doesn't confuse the 'Science' in 'Science Fiction' with 'Science' in a documentary
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create the final 'tags' column by combining everything
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new dataframe with just the essentials
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# --- STEP 3: TEXT PROCESSING & ML ENGINE ---
print("Training the recommendation engine...")

# Stemming: Convert words to their root form (e.g., loving -> love)
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

# Vectorization: Convert text tags into a matrix of numbers
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Calculate Cosine Similarity (distance between vectors)
similarity = cosine_similarity(vectors)

# --- STEP 4: EXPORT THE MODEL ---
print("Exporting model files...")
# We use pickle to save the dataframe and similarity matrix so our web app can load them instantly
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Success! Model trained and exported as 'movie_dict.pkl' and 'similarity.pkl'.")
