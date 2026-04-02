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

## 📂 Project Structure
├── model_artifacts/ # Saved model & metadata
├── vite-project/ # Frontend (React)
├── app.py # Flask API
├── main.py # Testing & experiments
├── save_model.py # Model training
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
└── README.md


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
