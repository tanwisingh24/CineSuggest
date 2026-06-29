"""
CineSuggest — a content-based movie recommendation system.

Loads a precomputed Bag-of-Words + cosine-similarity model (built from the
TMDB 5000 dataset in build_model.py) and lets a user pick a movie they like
to get five similar recommendations, with posters pulled from TMDB.
"""

import os
import pickle
from pathlib import Path

import requests
import streamlit as st

# --------------------------------------------------------------------------
# Page config & styling
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="CineSuggest",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path(__file__).parent / "data"
PLACEHOLDER_POSTER = "https://placehold.co/500x750/1c1c24/9b9bb0?text=No+Poster"

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #0e0e14 0%, #15151d 100%);
    }
    .block-container {
        padding-top: 2.5rem;
        max-width: 1100px;
    }
    h1, h2, h3 { color: #f2f2f7 !important; }

    .cs-header {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.1rem;
    }
    .cs-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #f2f2f7;
        letter-spacing: -0.02em;
    }
    .cs-title span { color: #e84545; }
    .cs-subtitle {
        color: #8b8b9e;
        font-size: 0.98rem;
        margin-top: 0;
        margin-bottom: 1.8rem;
    }

    div[data-baseweb="select"] > div {
        background-color: #1c1c26;
        border-color: #2c2c3a;
        border-radius: 10px;
    }

    .stButton > button {
        background-color: #e84545;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        transition: background-color 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #ff5757;
        color: white;
    }

    .movie-card {
        background-color: #1a1a23;
        border-radius: 14px;
        padding: 0.6rem;
        border: 1px solid #26262f;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .movie-card:hover {
        border-color: #e84545;
        transform: translateY(-3px);
    }
    .movie-card img {
        border-radius: 10px;
        width: 100%;
    }
    .movie-title {
        color: #e8e8f0;
        font-weight: 600;
        font-size: 0.92rem;
        margin-top: 0.5rem;
        line-height: 1.25;
        min-height: 2.4em;
    }
    .movie-meta {
        color: #8b8b9e;
        font-size: 0.78rem;
        margin-top: 0.15rem;
    }
    .genre-pill {
        display: inline-block;
        background-color: #2a2a36;
        color: #b7b7c9;
        font-size: 0.68rem;
        padding: 0.12rem 0.5rem;
        border-radius: 100px;
        margin-right: 0.3rem;
        margin-top: 0.4rem;
    }
    footer, #MainMenu { visibility: hidden; }
    .cs-footer {
        text-align: center;
        color: #5d5d6e;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-bottom: 1rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    with open(DATA_DIR / "movies.pkl", "rb") as f:
        movies = pickle.load(f)  # list of dicts
    with open(DATA_DIR / "similarity_topk.pkl", "rb") as f:
        sim = pickle.load(f)  # {"idx": [[...]], "score": [[...]], "k": int}
    titles = [m["title"] for m in movies]
    title_to_idx = {t: i for i, t in enumerate(titles)}
    return movies, sim, titles, title_to_idx


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster(movie_id: int) -> str:
    """Fetch a poster URL from TMDB. Falls back to a placeholder on any failure."""
    api_key = os.environ.get("TMDB_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("TMDB_API_KEY", "")
        except Exception:
            # st.secrets raises if no secrets.toml exists at all (not just
            # missing the key) -- that's expected when no key is configured.
            api_key = ""
    if not api_key:
        return PLACEHOLDER_POSTER
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        resp = requests.get(url, params={"api_key": api_key}, timeout=4)
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except (requests.RequestException, ValueError):
        pass
    return PLACEHOLDER_POSTER


def recommend(title: str, movies, sim: dict, title_to_idx: dict, n: int = 5):
    """Return up to n recommended movie record dicts."""
    movie_idx = title_to_idx.get(title)
    if movie_idx is None:
        return []
    neighbor_idxs = sim["idx"][movie_idx][:n]
    return [movies[int(j)] for j in neighbor_idxs]


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

movies, similarity, titles, title_to_idx = load_data()

st.markdown(
    '<div class="cs-header"><span class="cs-title">Cine<span>Suggest</span></span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="cs-subtitle">Pick a movie you like — get five more worth watching, '
    "based purely on plot, genre, cast and crew similarity.</p>",
    unsafe_allow_html=True,
)

col_select, col_btn = st.columns([5, 1])
with col_select:
    selected_movie = st.selectbox(
        "Search for a movie",
        titles,
        index=None,
        placeholder="Start typing a title, e.g. Inception, Avatar, The Dark Knight...",
        label_visibility="collapsed",
    )
with col_btn:
    go = st.button("Recommend", use_container_width=True)

if go and selected_movie:
    with st.spinner("Finding similar movies..."):
        recs = recommend(selected_movie, movies, similarity, title_to_idx, n=5)

    if not recs:
        st.warning("Couldn't find that title in the dataset. Try another one.")
    else:
        st.write("")
        cols = st.columns(5)
        for col, row in zip(cols, recs):
            poster_url = fetch_poster(int(row["movie_id"]))
            genres = row.get("genres") or []
            genre_html = "".join(
                f'<span class="genre-pill">{g}</span>' for g in genres[:2]
            )
            year = row.get("release_date", "")[:4] if row.get("release_date") else ""
            rating = row.get("vote_average")
            rating_str = f"⭐ {rating:.1f}" if rating else ""

            with col:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <img src="{poster_url}" alt="{row['title']}"/>
                        <div class="movie-title">{row['title']}</div>
                        <div class="movie-meta">{year} {('· ' + rating_str) if rating_str else ''}</div>
                        {genre_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
elif go and not selected_movie:
    st.info("Pick a movie first 🎞️")

st.markdown(
    '<div class="cs-footer">CineSuggest · content-based filtering '
    "(TF / Bag-of-Words + cosine similarity) on the TMDB 5000 dataset</div>",
    unsafe_allow_html=True,
)
