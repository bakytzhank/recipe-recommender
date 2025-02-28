import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load the cleaned dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_recipes.csv")
    
    # Handle missing or invalid data in 'NER_str'
    df['NER_str'] = df['NER_str'].fillna('')  # Replace missing values with empty strings
    df['NER_str'] = df['NER_str'].astype(str)  # Convert all values to strings
    
    # Clean the 'NER_str' column
    df['NER_str'] = df['NER_str'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())  # Remove extra spaces
    df['NER_str'] = df['NER_str'].str.lower()  # Convert to lowercase
    
    return df

df = load_data()

# Vectorize ingredients using HashingVectorizer
vectorizer = HashingVectorizer(stop_words='english', n_features=1000)
hash_matrix = vectorizer.fit_transform(df['NER_str'])

# Recommendation function
def recommend_recipes(user_ingredients, top_n=5):
    # Clean user input (remove measurements and special characters)
    user_ingredients = [re.sub(r"\d+\s?\w+\.?\s?", "", i).strip().lower() for i in user_ingredients]
    user_input_str = ' '.join(user_ingredients)
    
    # Vectorize user input
    user_input_vec = vectorizer.transform([user_input_str])
    
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(user_input_vec, hash_matrix)
    
    # Get top N similar recipes
    top_indices = similarity_scores.argsort()[0][-top_n:][::-1]
    return df.iloc[top_indices]

# Streamlit app
st.title("AI-Powered Recipe Recommender")
st.write("Enter the ingredients you have, and we'll recommend recipes!")

# User input
user_input = st.text_input("Enter ingredients (comma-separated):")
if user_input:
    user_ingredients = [x.strip() for x in user_input.split(',')]
    recommendations = recommend_recipes(user_ingredients)

    st.write("### Recommended Recipes:")
    for i, row in recommendations.iterrows():
        st.write(f"**{row['title']}**")
        st.write(f"**Ingredients:** {''.join(row['ingredients'])}")
        st.write(f"**Instructions:** {row['directions']}")
        st.write("---")