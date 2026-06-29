"""
CineSuggest — model build script.

Replicates the content-based filtering pipeline (TMDB 5000 dataset ->
genres/keywords/cast/crew parsing -> tags -> CountVectorizer (BoW) ->
cosine similarity) and saves compact pickle artifacts for the Streamlit app.

Usage:
    1. Download tmdb_5000_movies.csv and tmdb_5000_credits.csv from
       https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
       and place them in this folder.
    2. pip install -r requirements.txt
    3. python build_model.py
    4. Copy the generated movies.pkl and similarity_topk.pkl into ../data/

You only need to run this if you want to rebuild the model (e.g. with a
newer/larger dataset). The app ships with pre-built artifacts already in
../data/, so this step is optional for just running CineSuggest.
"""

import ast
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import nltk

    nltk.data.find("tokenizers/punkt")
except (ImportError, LookupError):
    pass

print("Loading datasets...")
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

print(f"movies: {movies.shape}, credits: {credits.shape}")

movies = movies.merge(credits, on="title")

# Keep only the columns we need downstream
movies = movies[
    [
        "movie_id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew",
        "vote_average",
        "vote_count",
        "release_date",
    ]
]

movies.dropna(subset=["overview", "genres", "keywords", "cast", "crew"], inplace=True)
movies.drop_duplicates(subset=["title"], inplace=True)
movies.reset_index(drop=True, inplace=True)


def convert(obj):
    """Extract 'name' field from a list of dicts encoded as a string."""
    try:
        return [i["name"] for i in ast.literal_eval(obj)]
    except (ValueError, SyntaxError):
        return []


def convert_cast(obj):
    """Keep only the first 3 cast members."""
    try:
        L = []
        for i, item in enumerate(ast.literal_eval(obj)):
            if i == 3:
                break
            L.append(item["name"])
        return L
    except (ValueError, SyntaxError):
        return []


def fetch_director(obj):
    try:
        for item in ast.literal_eval(obj):
            if item.get("job") == "Director":
                return [item["name"]]
        return []
    except (ValueError, SyntaxError):
        return []


print("Parsing genres / keywords / cast / crew...")
movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert_cast)
movies["crew"] = movies["crew"].apply(fetch_director)

# Keep a readable, space-preserved genre list for UI display before the
# tag-building step below strips spaces (needed so "Science Fiction"
# becomes one BoW token "sciencefiction" rather than two tokens).
genres_display = movies["genres"].apply(lambda x: list(x))

movies["overview"] = movies["overview"].apply(lambda x: x.split())

for col in ["genres", "keywords", "cast", "crew"]:
    movies[col] = movies[col].apply(lambda x: [str(i).replace(" ", "") for i in x])

movies["tags"] = (
    movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
)

new_df = movies[["movie_id", "title", "tags", "vote_average", "release_date"]].copy()
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())
new_df["genres_display"] = genres_display

print("Stemming...")
try:
    from nltk.stem.porter import PorterStemmer

    ps = PorterStemmer()

    def stem(text):
        return " ".join(ps.stem(w) for w in text.split())

    new_df["tags"] = new_df["tags"].apply(stem)
except Exception as e:
    print("NLTK stemming skipped:", e)

print("Vectorizing (Bag of Words, max_features=5000)...")
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()

print("Computing cosine similarity matrix...", vectors.shape)
similarity = cosine_similarity(vectors)

# Instead of persisting the full N x N matrix (which gets large fast and is
# mostly wasted, since the app only ever needs the top few neighbors of a
# movie) we precompute and store just the top-K most similar movies for
# each title. This is the standard way to ship a cosine-similarity
# recommender without shipping a huge matrix file.
K = 20
n = similarity.shape[0]
top_idx = np.zeros((n, K), dtype=np.int16)
top_score = np.zeros((n, K), dtype=np.float16)

for i in range(n):
    row = similarity[i]
    candidates = np.argpartition(-row, K + 1)[: K + 1]
    candidates = candidates[np.argsort(-row[candidates])]
    candidates = candidates[candidates != i][:K]
    top_idx[i] = candidates
    top_score[i] = row[candidates]

print("Saving artifacts...")
new_df_out = new_df[["movie_id", "title", "genres_display", "vote_average", "release_date"]].rename(
    columns={"genres_display": "genres"}
)

# Store as a plain list of dicts (not a DataFrame pickle) so the artifact
# loads reliably regardless of which pandas version is installed at
# deploy-time -- DataFrame pickles can break across pandas major/minor
# versions, plain Python objects do not.
records = new_df_out.to_dict(orient="records")
for r in records:
    r["movie_id"] = int(r["movie_id"])
    r["vote_average"] = float(r["vote_average"]) if pd.notna(r["vote_average"]) else None
    r["genres"] = list(r["genres"]) if r["genres"] is not None else []
    r["release_date"] = str(r["release_date"]) if pd.notna(r["release_date"]) else ""

import os

out_dir = os.path.join("..", "data")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "movies.pkl"), "wb") as f:
    pickle.dump(records, f, protocol=4)

with open(os.path.join(out_dir, "similarity_topk.pkl"), "wb") as f:
    pickle.dump(
        {"idx": top_idx.tolist(), "score": top_score.tolist(), "k": K},
        f,
        protocol=4,
    )

print("Done. Artifacts written to", os.path.abspath(out_dir))
print("movies.pkl rows:", len(records))
print("similarity_topk.pkl shape:", top_idx.shape)
