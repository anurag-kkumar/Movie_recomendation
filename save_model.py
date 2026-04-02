
import pandas as pd
import joblib
import numpy as np
import os



import ast
from sklearn.feature_extraction.text import TfidfVectorizer

movies_df = pd.read_csv("tmdb_5000_movies.csv")
credits_df = pd.read_csv("tmdb_5000_credits.csv")
df = movies_df.merge(credits_df, on='title')
df.drop(columns=['homepage', 'status', 'budget'], inplace=True, errors='ignore')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

poster_col = None
for c in ['poster_link', 'poster_path', 'poster', 'posterurl']:
    if c in df.columns:
        poster_col = c
        break

keep_cols = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
if poster_col:
    keep_cols.append(poster_col)
df = df[[c for c in keep_cols if c in df.columns]]

def extract_names(obj):
    try:
        return [i['name'].replace(" ", "") for i in ast.literal_eval(obj)]
    except Exception:
        return []

def extract_top_cast(obj):
    try:
        return [i['name'].replace(" ", "") for i in ast.literal_eval(obj)[:3]]
    except Exception:
        return []

def extract_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i.get('job') == 'Director':
                return [i['name'].replace(" ", "")]
    except Exception:
        pass
    return []

df['genres'] = df['genres'].apply(extract_names)
df['keywords'] = df['keywords'].apply(extract_names)
df['cast'] = df['cast'].apply(extract_top_cast)
df['crew'] = df['crew'].apply(extract_director)
df['overview'] = df['overview'].apply(lambda x: str(x).lower().split())

df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
df['tags'] = df['tags'].apply(lambda x: " ".join(x))
df.reset_index(drop=True, inplace=True)

tfidf = TfidfVectorizer(max_features=7000, stop_words='english')
vectors = tfidf.fit_transform(df['tags']) 

meta = pd.DataFrame({
    'title': df['title'].values,
    'genres': df['genres'].values
})
if poster_col:
    meta['poster'] = df[poster_col].fillna("").values
else:
    meta['poster'] = [""] * len(meta) 

os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(tfidf, "model_artifacts/tfidf.joblib")
joblib.dump(vectors, "model_artifacts/vectors_sparse.joblib")
meta.to_pickle("model_artifacts/meta.pkl")

print("Saved tfidf, vectors and meta to ./model_artifacts/")
print("Number of movies:", len(meta))
