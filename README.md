# CineSuggest 🎬

A content-based movie recommendation system. Pick a movie you like, and it suggests five others with similar genres, plot, cast and crew — no user accounts or ratings needed.

Built as a minor project for B.Tech CSE (AI/ML), and rebuilt here as a clean, deployable Streamlit app.

**Live demo:** _add your deployed link here once you deploy it_

## How it works

This isn't collaborative filtering (no "users who liked this also liked..."). It's pure content-based filtering:

1. Each movie's overview, genres, top 3 cast members, director and keywords are combined into one "tag" string.
2. Those tag strings are turned into vectors using a Bag-of-Words model (`CountVectorizer`, top 5000 words, English stop words removed).
3. Words are stemmed first (e.g. "fighting" → "fight") so similar words count as the same feature.
4. Cosine similarity is computed between every pair of movies.
5. For a selected movie, the 5 most similar movies (by cosine similarity) are recommended.

The dataset is the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) — about 4,800 movies after cleaning.

## Project structure

```
CineSuggest/
├── app.py                      # the Streamlit app
├── requirements.txt            # runtime dependencies (just streamlit + requests)
├── data/
│   ├── movies.pkl              # movie titles, genres, ratings (pre-built)
│   └── similarity_topk.pkl     # top-20 most similar movies per title (pre-built)
├── model_build/
│   ├── build_model.py          # rebuilds the two files above from raw CSVs
│   └── requirements.txt        # deps needed only to rebuild the model
└── .streamlit/
    ├── config.toml             # theme
    └── secrets.toml.example    # template for your TMDB API key
```

The app ships with the model already built, so you don't need to touch `model_build/` unless you want to regenerate it (e.g. with a bigger dataset).

## Running it locally

```bash
git clone https://github.com/<your-username>/CineSuggest.git
cd CineSuggest
pip install -r requirements.txt
streamlit run app.py
```

It'll open at `http://localhost:8501`.

### Adding movie posters (optional)

Without an API key, the app still works fine — it just shows a placeholder image instead of the real poster. To show real posters:

1. Get a free API key from [TMDB](https://www.themoviedb.org/settings/api).
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
3. Paste your key in:
   ```toml
   TMDB_API_KEY = "your_actual_key"
   ```

`secrets.toml` is in `.gitignore`, so it never gets committed.

## Deploying for free (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo, branch `main`, and set the main file to `app.py`.
4. If you want real posters, add `TMDB_API_KEY` under **Advanced settings → Secrets** in the same TOML format as above.
5. Deploy. First load takes a few seconds to spin up, then it's instant.

## Rebuilding the model (optional)

If you want to retrain on a different/bigger dataset:

```bash
cd model_build
pip install -r requirements.txt
# download tmdb_5000_movies.csv and tmdb_5000_credits.csv into this folder
# (from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
python build_model.py
```

This regenerates `data/movies.pkl` and `data/similarity_topk.pkl`.

One implementation note: instead of saving the full movie-by-movie similarity matrix (which would be a ~90MB file for this dataset), the build script only keeps each movie's top 20 nearest neighbours. Since the app only ever needs the top 5, this gives identical recommendations at a fraction of the file size — which is also generally how similarity-based recommenders are shipped in practice rather than carrying the full matrix around.

## Known limitations

- **Cold start**: a brand-new movie with no metadata can't be recommended.
- **Overspecialization**: recommendations skew toward very similar movies (e.g. sequels, same franchise) rather than introducing variety.
- **No personalization over time**: it doesn't learn from what you click — every recommendation is based only on the one movie you select.
- **No collaborative signal**: it has no idea what's currently popular or trending; it only looks at content.

These are discussed in more depth in the original project report this app is based on, along with directions for improving it (hybrid filtering, sentiment analysis on reviews, multimedia features, etc.).

## Tech stack

- Python, [Streamlit](https://streamlit.io/) for the UI
- pandas, scikit-learn (`CountVectorizer`, `cosine_similarity`) for the model
- NLTK (`PorterStemmer`) for text normalization
- [TMDB API](https://www.themoviedb.org/documentation/api) for posters

## License

MIT — see [LICENSE](LICENSE).
