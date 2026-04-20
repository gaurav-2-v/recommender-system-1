import streamlit as st
import pickle
import pandas as pd
import requests

# --- CONFIGURATION ---
# IMPORTANT: Replace 'YOUR_API_KEY' with your actual TMDB API key!
# You can get a free one by creating an account at https://www.themoviedb.org/settings/api
TMDB_API_KEY = "70ff7a308e9cc47cf562ba346bd4257e"

def fetch_poster(movie_id):
    """Fetches the official movie poster from the TMDB API."""
    if TMDB_API_KEY == "70ff7a308e9cc47cf562ba346bd4257e":
        # Fallback placeholder if user hasn't added their API key yet
        return "https://via.placeholder.com/500x750?text=No+API+Key"
        
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url).json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except Exception as e:
        return "https://via.placeholder.com/500x750?text=Poster+Not+Found"

def recommend(movie):
    """Finds the 5 most similar movies based on cosine similarity."""
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    # Sort distances to find the closest matches (excluding the movie itself at index 0)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_movies_posters = []
    
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        # Fetch poster from API
        recommended_movies_posters.append(fetch_poster(movie_id))
        
    return recommended_movies, recommended_movies_posters

# --- LOAD EXPORTED DATA ---
try:
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found! Please run 'data_preprocessing.py' first to generate the pickle files.")
    st.stop()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Movie Recommender", page_icon="🍿", layout="wide")

st.title("🍿 Movie Recommender System")
st.markdown("Select a movie you like, and we'll recommend 5 similar movies!")

# Dropdown for selecting a movie
selected_movie_name = st.selectbox(
    'Search for a movie:',
    movies['title'].values
)

# Button to trigger recommendation
if st.button('Recommend'):
    with st.spinner('Finding recommendations...'):
        names, posters = recommend(selected_movie_name)
        
        # Display recommendations in 5 columns
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.text(names[i])
                st.image(posters[i])