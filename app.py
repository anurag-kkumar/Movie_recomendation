from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
CORS(app) 

tfidf = joblib.load("model_artifacts/tfidf.joblib")
vectors = joblib.load("model_artifacts/vectors_sparse.joblib")  
meta = pd.read_pickle("model_artifacts/meta.pkl")

title_to_index = {t.lower(): i for i, t in enumerate(meta['title'].values)}

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data or 'title' not in data:
        return jsonify({"error": "Please send JSON with 'title' field."}), 400

    title = data['title'].strip().lower()
    if title not in title_to_index:
        # try fuzzy fallback: find closest title by substring match
        candidates = [t for t in title_to_index.keys() if title in t]
        if candidates:
            chosen = candidates[0]
            idx = title_to_index[chosen]
        else:
            return jsonify({"error": f"Movie '{data['title']}' not found."}), 404
    else:
        idx = title_to_index[title]

    movie_vector = vectors[idx]
    sims = cosine_similarity(movie_vector, vectors).flatten() 

    k = 6  
    top_idx = np.argsort(-sims)[:k] 
    results = []
    for i in top_idx:
        if i == idx:
            continue
        results.append({
            "title": str(meta.iloc[i]['title']),
            "poster": str(meta.iloc[i]['poster']) if meta.iloc[i]['poster'] else "",
            "genres": meta.iloc[i]['genres'],
            "score": float(sims[i])
        })
        if len(results) >= 5:
            break

    response = {
        "query": meta.iloc[idx]['title'],
        "results": results
    }
    return jsonify(response), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
