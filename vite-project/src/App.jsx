// src/App.js
import React, { useState } from "react";
import "./index.css";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSearch(e) {
    e.preventDefault();
    setError(null);
    setResults([]);
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("http://localhost:5000/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: query })
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Error fetching recommendations");
      } else {
        setResults(data.results || []);
      }
    } catch (err) {
      setError("Network error: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  // helper to show poster or placeholder
  const posterOrPlaceholder = (poster) =>
    poster && poster.length > 8 ? poster : "https://via.placeholder.com/300x450?text=No+Poster";

  return (
    <div className="app">
      <header>
        <h1>Movie Recommender</h1>
        <p>Type a movie name and get top 5 similar movies</p>
      </header>

      <form className="search-form" onSubmit={handleSearch}>
        <input
          type="text"
          value={query}
          placeholder="e.g. The Matrix, Avatar, Titanic..."
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      <main>
        <div className="grid">
          {results.map((m) => (
            <div className="card" key={m.title}>
              <img src={posterOrPlaceholder(m.poster)} alt={m.title} />
              <div className="meta">
                <h3>{m.title}</h3>
                <p className="score">Score: {m.score.toFixed(3)}</p>
                <p className="genres">{Array.isArray(m.genres) ? m.genres.join(", ") : m.genres}</p>
              </div>
            </div>
          ))}
        </div>
      </main>

      <footer>
      </footer>
    </div>
  );
}

export default App;
