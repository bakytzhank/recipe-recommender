# 🍳 AI-Powered Recipe Recommender App

This is a **Streamlit-based web app** that recommends recipes based on the ingredients you have. It uses the [MealDB API](https://www.themealdb.com/api.php) to fetch recipes and ranks them using **ingredient-based similarity** powered by **TF-IDF** and **cosine similarity**.

---

## Features

- **Ingredient-Based Search**: Enter the ingredients you have, and the app will find matching recipes.
- **AI-Powered Recommendations**: Recipes are ranked based on how closely their ingredients match your input.
- **Recipe Details**: View detailed instructions, ingredients, and thumbnails for each recipe.
- **Simple and Intuitive**: No complex setup—just enter ingredients and get recommendations!

---

## How It Works

1. The app fetches recipes from the **MealDB API** that include the ingredients you provide.
2. It uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert recipe ingredients into numerical vectors.
3. **Cosine similarity** is used to rank recipes based on how closely their ingredients match your input.
4. The top 5 recipes are displayed with their details (ingredients, instructions, and thumbnails).

---

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/bakytzhank/recipe-recommender.git
    ```
2. Navigate to the project folder:
    ```
    cd recipe-recommender-app
    ```
3. Install the dependencies:
    ```
    pip install -r app/requirements.txt
    ```

---

## Running the App Locally

1. Start the Streamlit app:
    ```
    streamlit run app/app.py
    ```

2. Open your browser and go to `http://localhost:8501`.

3. Enter the ingredients you have (e.g., `chicken, rice, tomato`) and click **Find Recipes**.

4. View the recommended recipes along with their details.

---

## Technologies Used

- **Streamlit**: For building the web app.
- **MealDB API**: For fetching recipe data.
- **Scikit-learn**: For TF-IDF and cosine similarity.
- **Pandas**: For data manipulation.
- **Requests**: For making API calls.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---