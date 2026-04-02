# 🎬 Movie Recommendation System

A **Machine Learning-based Movie Recommendation System** that suggests similar movies based on a given title using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 🚀 Features
- 🔍 Search movies by title  
- 🎯 Get top 5 similar recommendations  
- ⚡ Fast Flask API  
- 🧠 Content-based filtering (overview, genres, cast, crew)  
- 🖼️ Poster support (if available)  
- 🎨 Optional frontend (Vite + React)

---

## 🛠️ Tech Stack
- **Backend:** Python, Flask, Scikit-learn, Pandas, NumPy  
- **Frontend:** React (Vite)  
- **ML:** TF-IDF + Cosine Similarity  

---


### 📁 Project Structure

- **model_artifacts/**  
  - Contains saved model files  
  - Includes similarity matrix and vectorizer  

- **vite-project/**  
  - Frontend built using React + Vite  

- **app.py**  
  - Flask API (backend server)  

- **main.py**  
  - Used for testing and experimentation  

- **save_model.py**  
  - Script for training and saving the model  

- **tmdb_5000_movies.csv**  
  - Dataset containing movie details  

- **tmdb_5000_credits.csv**  
  - Dataset containing movie credits  

- **README.md**  
  - Project documentation  


---

## ⚙️ How It Works
1. Combine movie features (overview, genres, keywords, cast, crew)  
2. Convert text into vectors using **TF-IDF**  
3. Compute similarity using **cosine similarity**  
4. Return top 5 similar movies  

---

## 🧪 API

- Endpoint: POST /recommend  
- Input: Movie title  
- Output: Top 5 similar movies with details  
