"""
Single-query prediction function.

predict("tt33022333")            → uses IMDb ID directly
predict("Chainsaw Man")          → resolves title via IMDb datasets
predict("Chainsaw Man", year=2022)  → disambiguates by year
"""
import sys
import numpy as np
import lightgbm as lgb

from mrp.features import get_movie_features
from mrp.feature_builder import FeatureBuilder
from mrp.imdb_data import get_instance as get_imdb
from mrp import omdb
from mrp.config import MODELS_DIR


_model = None
_builder = None


def _load_model():
    global _model, _builder
    if _model is None:
        model_path = MODELS_DIR / "model.lgb"
        builder_path = MODELS_DIR / "feature_builder.pkl"
        if not model_path.exists() or not builder_path.exists():
            raise FileNotFoundError(
                "Model not found. Run training first:\n"
                "  python -m mrp train"
            )
        _model = lgb.Booster(model_file=str(model_path))
        _builder = FeatureBuilder.load(str(builder_path))
    return _model, _builder


def resolve_title(query, year=None):
    """Resolve a title string (or pass through an IMDb ID). Returns (imdb_id, omdb_data)."""
    # Already an IMDb ID?
    if query.lower().startswith("tt") and query[2:].isdigit():
        return query, None

    # Use OMDb for instant title search (costs 1 API call, saves 40 seconds)
    print(f"  Searching OMDb for '{query}'...")
    tconst, omdb_data = omdb.search_by_title(query, year)
    if tconst:
        return tconst, omdb_data

    # Fall back to local IMDb datasets if OMDb fails
    print("  Not found on OMDb. Loading local IMDb data (this takes 30s)...")
    imdb = get_imdb()
    if imdb.basics is None:
        imdb._load_basics()
        imdb._load_ratings()
        imdb.loaded = True

    return imdb.search_title(query, year), None


def predict(query, year=None):
    """
    Predict personal rating (1–10) for a movie.
    """
    # 1. Resolve to IMDb ID
    imdb_id, prefetched_omdb = resolve_title(query, year)
    if imdb_id is None:
        print(f"✗ Could not resolve '{query}' to an IMDb ID")
        return None

    # 2. Fetch features. Pass along any OMDb data we already fetched!
    print(f"  Fetching live features for {imdb_id}...")
    features = get_movie_features(imdb_id, prefetched_omdb=prefetched_omdb)
    if features is None or features.get("status") != "ok":
        print(f"✗ Could not fetch features for {imdb_id}")
        return None

    # 3. Load model + feature builder
    model, builder = _load_model()

    # 4. Build feature vector
    vec = builder.transform(features)
    pred = model.predict(vec.reshape(1, -1))[0]
    pred = float(np.clip(pred, 1.0, 10.0))

    return {
        "imdb_id": imdb_id,
        "title": features.get("title", ""),
        "year": features.get("year"),
        "predicted_rating": round(pred, 2),
        "imdb_public_rating": features.get("imdb_rating"),
        "genres": features.get("genres", []),
        "directors": features.get("directors", []),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Predict a movie rating")
    p.add_argument("query", help="Movie title or IMDb ID (tt...)")
    p.add_argument("--year", type=int, default=None)
    args = p.parse_args()

    result = predict(args.query, args.year)
    if result:
        print()
        print(f"  Title:    {result['title']}" + (f" ({result['year']})" if result['year'] else ""))
        print(f"  IMDb ID:  {result['imdb_id']}")
        print(f"  Genres:   {', '.join(result['genres'])}")
        print(f"  Director: {', '.join(result['directors'])}")
        print(f"  IMDb avg: {result['imdb_public_rating']}")
        print(f"  ────────────────────────────────")
        print(f"  Predicted rating: {result['predicted_rating']:.1f}/10")
        print()
    else:
        sys.exit(1)