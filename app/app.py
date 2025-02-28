import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title of the app
st.title("🍳 AI-Powered Recipe Recommender")

# User input for ingredients
ingredients = st.text_input("Enter ingredients you have (comma-separated):", "chicken, rice, tomato")

# Function to fetch recipes from MealDB API
def fetch_recipes(ingredients):
    # Split ingredients into a list
    ingredient_list = [ingredient.strip() for ingredient in ingredients.split(",")]
    
    # Fetch recipes from MealDB API
    recipes = []
    for ingredient in ingredient_list:
        response = requests.get(f"https://www.themealdb.com/api/json/v1/1/filter.php?i={ingredient}")
        if response.status_code == 200:
            data = response.json()
            if data["meals"]:
                recipes.extend(data["meals"])
    
    # Remove duplicate recipes
    unique_recipes = []
    seen_ids = set()
    for recipe in recipes:
        if recipe["idMeal"] not in seen_ids:
            unique_recipes.append(recipe)
            seen_ids.add(recipe["idMeal"])
    
    return unique_recipes

# Function to get recipe details by ID
def get_recipe_details(recipe_id):
    response = requests.get(f"https://www.themealdb.com/api/json/v1/1/lookup.php?i={recipe_id}")
    if response.status_code == 200:
        data = response.json()
        return data["meals"][0]
    return None

# Function to rank recipes based on ingredient similarity
def rank_recipes(recipes, user_ingredients):
    # Create a DataFrame for recipes
    recipe_data = []
    for recipe in recipes:
        details = get_recipe_details(recipe["idMeal"])
        if details:
            recipe_data.append({
                "id": details["idMeal"],
                "name": details["strMeal"],
                "ingredients": ", ".join([details[f"strIngredient{i}"] for i in range(1, 21) if details[f"strIngredient{i}"]])
            })
    df = pd.DataFrame(recipe_data)
    
    # Compute TF-IDF vectors for recipe ingredients
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['ingredients'])
    
    # Compute TF-IDF vector for user ingredients
    user_tfidf = vectorizer.transform([user_ingredients])
    
    # Compute cosine similarity between user ingredients and recipe ingredients
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Add similarity scores to the DataFrame
    df['similarity'] = cosine_sim
    
    # Sort recipes by similarity score
    df = df.sort_values(by='similarity', ascending=False)
    
    return df

# Display recipes based on user input
if st.button("Find Recipes"):
    recipes = fetch_recipes(ingredients)
    
    if not recipes:
        st.warning("No recipes found. Try different ingredients!")
    else:
        # Rank recipes based on ingredient similarity
        ranked_recipes = rank_recipes(recipes, ingredients)
        
        # Display top 5 recipes with details
        st.write("### Recommended Recipes")
        for _, row in ranked_recipes.head(5).iterrows():
            st.write(f"#### {row['name']}")
            st.write(f"**Similarity Score**: {row['similarity']:.2f}")
            details = get_recipe_details(row['id'])
            if details:
                st.image(details["strMealThumb"], width=200)
                
                # Display ingredients
                st.write("##### Ingredients:")
                for i in range(1, 21):
                    ingredient = details.get(f"strIngredient{i}")
                    measure = details.get(f"strMeasure{i}")
                    if ingredient and measure:
                        st.write(f"- {ingredient}: {measure}")
                
                # Display instructions
                st.write("##### Instructions:")
                st.write(details["strInstructions"])
                
                # Add a separator between recipes
                st.write("---")