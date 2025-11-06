
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

movies_df = pd.read_csv("tmdb_5000_movies.csv")
credits_df = pd.read_csv("tmdb_5000_credits.csv")

df = movies_df.merge(credits_df, on='title')

df.drop(columns=['homepage', 'status', 'budget'], inplace=True, errors='ignore')

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)


df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

def extract_names(obj):
    """Extract list of 'name' values from a JSON-like string."""
    names = []
    for i in ast.literal_eval(obj):
        names.append(i['name'])
    return names

def extract_top_cast(obj):
    """Extract top 3 cast members."""
    cast = []
    for i in ast.literal_eval(obj)[:3]:
        cast.append(i['name'])
    return cast

def extract_director(obj):
    """Extract the director name."""
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

df['genres'] = df['genres'].apply(extract_names)
df['keywords'] = df['keywords'].apply(extract_names)
df['cast'] = df['cast'].apply(extract_top_cast)
df['crew'] = df['crew'].apply(extract_director)
df['overview'] = df['overview'].apply(lambda x: x.split())

# Combine features into one column
df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
df['tags'] = df['tags'].apply(lambda x: " ".join(x).lower())

df.reset_index(drop=True, inplace=True)


cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

print("Original feature dimensions:", vectors.shape)

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

print("Reduced feature dimensions:", reduced_vectors.shape)


plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    alpha=0.6,
    s=40,
    edgecolor='k'
)

plt.title(" Movie Clusters (PCA - 2D Visualization)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

sample_titles = df.sample(10, random_state=42)
for i, row in sample_titles.iterrows():
    plt.text(
        reduced_vectors[i, 0],
        reduced_vectors[i, 1],
        row['title'],
        fontsize=8
    )

plt.show()


knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(vectors)

def recommend_movies(movie_title, n_recommendations=5):
    """Recommend top N similar movies using cosine similarity."""
    if movie_title not in df['title'].values:
        return f" Movie '{movie_title}' not found in the dataset."

    movie_idx = df[df['title'] == movie_title].index[0]
    movie_vector = vectors[movie_idx]

    distances, indices = knn_model.kneighbors([movie_vector], n_neighbors=n_recommendations + 1)

    similar_movies = []
    for i in range(1, len(indices[0])): 
        idx = indices[0][i]
        similar_movies.append((df.iloc[idx]['title'], 1 - distances[0][i]))  

    return similar_movies


test_movies = ["Avatar", "The Dark Knight", "The Notebook"]

for movie in test_movies:
    print(f"\n Top 5 movies similar to '{movie}':\n")
    results = recommend_movies(movie)
    if isinstance(results, str):
        print(results)
    else:
        for title, score in results:
            print(f"  → {title} (Similarity: {score:.3f})")



def evaluate_accuracy(n_samples=10, n_recommendations=5):
    """
    Evaluate how accurate recommendations are
    based on genre overlap between recommended movies and the input movie.
    """
    sampled_movies = df.sample(n_samples, random_state=42)
    genre_overlap_scores = []
    cosine_sim_scores = []

    for _, row in sampled_movies.iterrows():
        movie_title = row['title']
        movie_genres = set(row['genres'])
        
        # Get recommendations
        recommendations = recommend_movies(movie_title, n_recommendations=n_recommendations)
        if isinstance(recommendations, str):
            continue
        
        # Calculate genre overlap and cosine similarity
        overlap_total = 0
        cos_total = 0
        
        movie_idx = df[df['title'] == movie_title].index[0]
        movie_vector = vectors[movie_idx].reshape(1, -1)
        
        for rec_title, sim_score in recommendations:
            rec_genres = set(df[df['title'] == rec_title]['genres'].values[0])
            overlap = len(movie_genres.intersection(rec_genres)) / max(len(movie_genres), 1)
            overlap_total += overlap
            cos_total += sim_score 

        genre_overlap_scores.append(overlap_total / n_recommendations)
        cosine_sim_scores.append(cos_total / n_recommendations)
    
    mean_genre_overlap = np.mean(genre_overlap_scores)
    mean_cosine_similarity = np.mean(cosine_sim_scores)
    
    print(" Model Evaluation Results:")
    print(f"Average Genre Overlap Accuracy: {mean_genre_overlap * 100:.2f}%")
    print(f"Average Cosine Similarity: {mean_cosine_similarity:.3f}")
    return mean_genre_overlap, mean_cosine_similarity

evaluate_accuracy(n_samples=10, n_recommendations=5)
