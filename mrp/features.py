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

# Fields that only an OMDb call provides. A cached entry missing any of
# these (e.g. fetched before enrichment existed) is considered incomplete.
_ENRICHMENT_KEYS = (
    "metascore", "rated", "countries", "languages",
    "awards_wins", "awards_nominations", "box_office",
)

# Re-fetch OMDb data when the cached entry is older than this.
OMDB_MAX_AGE_DAYS = 1


def _is_stale(cached):
    """True if the cached entry's fetched_at is older than OMDB_MAX_AGE_DAYS."""
    fetched_at = (cached or {}).get("fetched_at") or ""
    try:
        from datetime import datetime, timedelta
        age = datetime.now() - datetime.fromisoformat(fetched_at)
        return age > timedelta(days=OMDB_MAX_AGE_DAYS)
    except ValueError:
        return False


def get_movie_features(imdb_id, force_refresh=False, prefetched_omdb=None):
    """
    Fetch and cache all features for *imdb_id*.
    """
    cache = get_cache()

    # ── Cache hit (complete entry) ─────────────────────────────────────────
    if not force_refresh:
        cached = cache.get(imdb_id)
        if (cached and cached.get("status") == "ok" and cached.get("plot")
                and all(k in cached for k in _ENRICHMENT_KEYS)):
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

    # ── 2. OMDb for plot + any missing/stale structured/enrichment fields ──
    plot_changed = False
    need_omdb = (
        not features.get("plot")
        or not features.get("cast")
        or not features.get("title")
        or any(k not in features for k in _ENRICHMENT_KEYS)
        or _is_stale(features)
    )
    if need_omdb:
        # Use prefetched data if we have it (from a title search), otherwise fetch it
        old_plot = features.get("plot")
        omdb_data = prefetched_omdb if prefetched_omdb else omdb.fetch(imdb_id)
        if omdb_data:
            # On a staleness refresh, overwrite existing values so updated
            # ratings/votes/etc. actually propagate into the cache.
            overwrite = _is_stale(features)
            for k, v in omdb_data.items():
                if v and (overwrite or not features.get(k)):
                    features[k] = v
            if omdb_data.get("plot"):
                features["plot"] = omdb_data["plot"]
            plot_changed = features.get("plot") != old_plot
            # Mark enrichment as attempted so backfill doesn't re-fetch
            # this movie on every run.
            features["enriched"] = True

    # ── 3. Wikipedia fallback for plot ─────────────────────────────────────
    if not features.get("plot") and features.get("title"):
        wiki = wikipedia_plot.fetch_plot(features["title"], features.get("year"))
        if wiki:
            features["plot"] = wiki

    # ── 4. Embedding (recompute if plot text changed) ──────────────────────
    if features.get("embedding") is None or force_refresh or plot_changed:
        features["embedding"] = embeddings.embed(features.get("plot", ""))

    # ── 5. Status & cache ──────────────────────────────────────────────────
    features["status"] = "ok" if features.get("title") else "error"
    cache.set(imdb_id, features)

    return features