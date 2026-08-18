"""
Converts raw feature dicts (from get_movie_features) into a fixed-size
numeric matrix for LightGBM.

Fitted once during training, then serialised and reloaded at inference
time so that train and serve produce identical column ordering.
"""
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
from mrp.config import (
    TOP_N_DIRECTORS,
    TOP_N_ACTORS,
    PCA_COMPONENTS,
    EMBEDDING_DIM,
)


class FeatureBuilder:
    """Learns the feature schema from training data and applies it consistently."""

    def __init__(self):
        self.genres = []
        self.title_types = []
        self.top_directors = []
        self.top_actors = []
        self.pca = None
        self.feature_names_ = []

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(self, features_list):
        """Learn vocabulary + PCA from a list of feature dicts."""
        # genres
        genre_set = set()
        for f in features_list:
            genre_set.update(f.get("genres", []))
        self.genres = sorted(genre_set)

        # title types
        type_set = set()
        for f in features_list:
            t = f.get("title_type")
            if t:
                type_set.add(t)
        self.title_types = sorted(type_set)

        # top-N directors by frequency
        dir_counts = Counter()
        for f in features_list:
            for d in f.get("directors", []):
                dir_counts[d] += 1
        self.top_directors = [d for d, _ in dir_counts.most_common(TOP_N_DIRECTORS)]

        # top-N actors by frequency
        act_counts = Counter()
        for f in features_list:
            for a in f.get("cast", []):
                act_counts[a] += 1
        self.top_actors = [a for a, _ in act_counts.most_common(TOP_N_ACTORS)]

        # PCA on embeddings
        emb_matrix = np.array(
            [
                f.get("embedding")
                if f.get("embedding") is not None
                else np.zeros(EMBEDDING_DIM, dtype=np.float32)
                for f in features_list
            ],
            dtype=np.float32,
        )
        n_comp = min(
            PCA_COMPONENTS,
            emb_matrix.shape[0],
            emb_matrix.shape[1],
        )
        if n_comp > 0:
            self.pca = PCA(n_components=n_comp, random_state=42)
            self.pca.fit(emb_matrix)

        # determine feature name order from one sample
        sample = self._transform_raw(features_list[0])
        self.feature_names_ = list(sample.keys())

        return self

    # ── Transform ──────────────────────────────────────────────────────────

    def transform(self, features):
        """
        Convert a single feature dict → 1-D numpy array in training column order.
        Unknown genres/directors/actors are silently dropped (value 0).
        """
        vec = self._transform_raw(features)
        return np.array([vec.get(name, 0.0) for name in self.feature_names_],
                        dtype=np.float64)

    def transform_batch(self, features_list):
        """Convert a list of feature dicts → pandas DataFrame."""
        rows = [self.transform(f) for f in features_list]
        return pd.DataFrame(rows, columns=self.feature_names_)

    def _transform_raw(self, f):
        """Build the full feature dict (no column-order enforcement)."""
        vec = {}

        # ── numeric ────────────────────────────────────────────────────────
        vec["imdb_rating"] = f.get("imdb_rating") if f.get("imdb_rating") is not None else np.nan
        votes = f.get("imdb_votes") or 0
        vec["log_votes"] = float(np.log1p(votes))
        vec["runtime"] = f.get("runtime") if f.get("runtime") is not None else np.nan
        ## optional: keep year as feature? 
        # vec["year"] = f.get("year") if f.get("year") is not None else np.nan
        vec["num_genres"] = len(f.get("genres", []))
        vec["num_directors"] = len(f.get("directors", []))
        vec["num_cast"] = len(f.get("cast", []))

        # ── genre flags ────────────────────────────────────────────────────
        movie_genres = set(f.get("genres", []))
        for g in self.genres:
            vec[f"genre__{g}"] = 1.0 if g in movie_genres else 0.0

        # ── title-type flags ───────────────────────────────────────────────
        for t in self.title_types:
            vec[f"type__{t}"] = 1.0 if f.get("title_type") == t else 0.0

        # ── director flags ─────────────────────────────────────────────────
        movie_dirs = set(f.get("directors", []))
        for d in self.top_directors:
            vec[f"dir__{d}"] = 1.0 if d in movie_dirs else 0.0

        # ── actor flags ────────────────────────────────────────────────────
        movie_cast = set(f.get("cast", []))
        for a in self.top_actors:
            vec[f"actor__{a}"] = 1.0 if a in movie_cast else 0.0

        # ── plot embedding (PCA-reduced) ───────────────────────────────────
        emb = f.get("embedding")
        if emb is not None and self.pca is not None:
            reduced = self.pca.transform(emb.reshape(1, -1).astype(np.float32))
            for i, val in enumerate(reduced[0]):
                vec[f"emb__{i}"] = float(val)
        elif self.pca is not None:
            for i in range(self.pca.n_components_):
                vec[f"emb__{i}"] = 0.0

        return vec

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "genres": self.genres,
                    "title_types": self.title_types,
                    "top_directors": self.top_directors,
                    "top_actors": self.top_actors,
                    "pca": self.pca,
                    "feature_names_": self.feature_names_,
                },
                fh,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        obj = cls()
        obj.genres = d["genres"]
        obj.title_types = d["title_types"]
        obj.top_directors = d["top_directors"]
        obj.top_actors = d["top_actors"]
        obj.pca = d["pca"]
        obj.feature_names_ = d["feature_names_"]
        return obj