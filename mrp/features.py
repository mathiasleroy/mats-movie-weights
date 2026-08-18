"""
The single shared feature-fetch function.

get_movie_features(imdb_id) is called identically during backfill
(training data assembly) and at inference time.  This is the
train/serve consistency hard-requirement.
"""
import numpy as np

from mrp.cache import get_cache
from mrp.imdb_data import get_instance as get_imdb
from mrp import omdb
from mrp import wikipedia_plot
from mrp import embeddings


def get_movie_features(imdb_id, force_refresh=False, prefetched_omdb=None):
    """
    Fetch and cache all features for *imdb_id*.
    """
    cache = get_cache()

    # ── Cache hit (complete entry) ─────────────────────────────────────────
    if not force_refresh:
        cached = cache.get(imdb_id)
        if cached and cached.get("status") == "ok" and cached.get("plot"):
            return cached

    # Start from partial cache if we have one
    features = cached if cached else {}
    features.setdefault("imdb_id", imdb_id)
    features.setdefault("title", "")
    features.setdefault("original_title", "")
    features.setdefault("title_type", "")
    features.setdefault("year", None)
    features.setdefault("runtime", None)
    features.setdefault("genres", [])
    features.setdefault("directors", [])
    features.setdefault("cast", [])
    features.setdefault("imdb_rating", None)
    features.setdefault("imdb_votes", None)
    features.setdefault("plot", "")

    # ── 1. Structured features from IMDb datasets ──────────────────────────
    imdb = get_imdb()
    if imdb.loaded:
        imdb_data = imdb.lookup(imdb_id)
        if imdb_data:
            for k, v in imdb_data.items():
                if not features.get(k) and v:
                    features[k] = v

    # ── 2. OMDb for plot + any missing structured fields ───────────────────
    need_omdb = (
        not features.get("plot")
        or not features.get("cast")
        or not features.get("title")
    )
    if need_omdb:
        # Use prefetched data if we have it (from a title search), otherwise fetch it
        omdb_data = prefetched_omdb if prefetched_omdb else omdb.fetch(imdb_id)
        if omdb_data:
            for k, v in omdb_data.items():
                if not features.get(k) and v:
                    features[k] = v
            if omdb_data.get("plot"):
                features["plot"] = omdb_data["plot"]

    # ── 3. Wikipedia fallback for plot ─────────────────────────────────────
    if not features.get("plot") and features.get("title"):
        wiki = wikipedia_plot.fetch_plot(features["title"], features.get("year"))
        if wiki:
            features["plot"] = wiki

    # ── 4. Embedding ───────────────────────────────────────────────────────
    if features.get("embedding") is None or force_refresh:
        features["embedding"] = embeddings.embed(features.get("plot", ""))

    # ── 5. Status & cache ──────────────────────────────────────────────────
    features["status"] = "ok" if features.get("title") else "error"
    cache.set(imdb_id, features)

    return features