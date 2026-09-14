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
    PCA_COMPONENTS,
    EMBEDDING_DIM,
)

# smoothing strength for target encoding (higher = more shrink to global mean)
TE_SMOOTHING = 15
TOP_N_COUNTRIES = 10
TOP_N_LANGUAGES = 8
TOP_N_RATED = 8


class FeatureBuilder:
    """Learns the feature schema from training data and applies it consistently."""

    def __init__(self):
        self.genres = []
        self.title_types = []
        self.dir_te = {}        # director -> smoothed personal avg rating
        self.cast_te = {}       # actor -> smoothed personal avg rating
        self.writer_te = {}     # writer -> smoothed personal avg rating
        self.countries = []     # top-N country names
        self.languages = []     # top-N language names
        self.rated = []         # rating certificates (R, PG-13, ...)
        self.global_mean = 6.25
        self.pca = None
        self.feature_names_ = []

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(self, features_list, y):
        """
        Learn vocabulary + PCA + target encodings from training data.

        Target encoding replaces the old one-hot director/actor flags
        (which LightGBM almost never split on) with each person's
        smoothed personal average rating:
            score(p) = (sum(ratings of p's movies) + K * global_mean) / (n_p + K)
        """
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

        # target encodings for directors, cast & writers
        self.global_mean = float(np.mean(y))
        k = TE_SMOOTHING
        sums = {"dir": {}, "cast": {}, "writer": {}}
        ns = {"dir": Counter(), "cast": Counter(), "writer": Counter()}
        keys = {"dir": "directors", "cast": "cast", "writer": "writers"}
        for f, r in zip(features_list, y):
            for grp, key in keys.items():
                for p in f.get(key, []) or []:
                    sums[grp][p] = sums[grp].get(p, 0.0) + r
                    ns[grp][p] += 1
        def _te(grp):
            return {p: (sums[grp][p] + k * self.global_mean) / (ns[grp][p] + k)
                    for p in ns[grp]}
        self.dir_te = _te("dir")
        self.cast_te = _te("cast")
        self.writer_te = _te("writer")

        # top-N vocabularies for categorical OMDb fields
        country_counts, lang_counts, rated_counts = Counter(), Counter(), Counter()
        for f in features_list:
            for c in f.get("countries", []) or []:
                country_counts[c] += 1
            for l in f.get("languages", []) or []:
                lang_counts[l] += 1
            if f.get("rated"):
                rated_counts[f["rated"]] += 1
        self.countries = [c for c, _ in country_counts.most_common(TOP_N_COUNTRIES)]
        self.languages = [l for l, _ in lang_counts.most_common(TOP_N_LANGUAGES)]
        self.rated = [r for r, _ in rated_counts.most_common(TOP_N_RATED)]

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

        # ── genre flags ────────────────────────────────────────────────────
        movie_genres = set(f.get("genres", []))
        for g in self.genres:
            vec[f"genre__{g}"] = 1.0 if g in movie_genres else 0.0

        # ── title-type flags ───────────────────────────────────────────────
        for t in self.title_types:
            vec[f"type__{t}"] = 1.0 if f.get("title_type") == t else 0.0

        # ── director/cast/writer target encodings ──────────────────────────
        gm = self.global_mean
        dirs = [self.dir_te.get(d, gm) for d in f.get("directors", [])]
        vec["dir_te"] = float(np.mean(dirs)) if dirs else gm
        cast_scores = [self.cast_te.get(a, gm) for a in f.get("cast", [])]
        vec["cast_te"] = float(np.mean(cast_scores)) if cast_scores else gm
        # vec["cast_te_max"] = float(max(cast_scores, default=gm))
        # vec["cast_te_min"] = float(min(cast_scores, default=gm))
        writer_scores = [self.writer_te.get(w, gm) for w in f.get("writers", [])]
        vec["writer_te"] = float(np.mean(writer_scores)) if writer_scores else gm
        # vec["writer_te_max"] = float(max(writer_scores, default=gm))
        # vec["writer_te_min"] = float(min(writer_scores, default=gm))

        # ── OMDb enrichment ────────────────────────────────────────────────
        ms = f.get("metascore")
        vec["metascore"] = float(ms) if ms is not None else np.nan
        bo = f.get("box_office")
        vec["log_box_office"] = float(np.log1p(bo)) if bo else np.nan
        aw = f.get("awards_wins")
        vec["log_awards_wins"] = float(np.log1p(aw)) if aw else 0.0
        an = f.get("awards_nominations")
        vec["log_awards_noms"] = float(np.log1p(an)) if an else 0.0
        for r in self.rated:
            vec[f"rated__{r}"] = 1.0 if f.get("rated") == r else 0.0
        movie_countries = set(f.get("countries", []) or [])
        for c in self.countries:
            vec[f"country__{c}"] = 1.0 if c in movie_countries else 0.0
        movie_langs = set(f.get("languages", []) or [])
        for l in self.languages:
            vec[f"lang__{l}"] = 1.0 if l in movie_langs else 0.0

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
                    "dir_te": self.dir_te,
                    "cast_te": self.cast_te,
                    "writer_te": self.writer_te,
                    "countries": self.countries,
                    "languages": self.languages,
                    "rated": self.rated,
                    "global_mean": self.global_mean,
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
        obj.dir_te = d.get("dir_te", {})
        obj.cast_te = d.get("cast_te", {})
        obj.writer_te = d.get("writer_te", {})
        obj.countries = d.get("countries", [])
        obj.languages = d.get("languages", [])
        obj.rated = d.get("rated", [])
        obj.global_mean = d.get("global_mean", 6.25)
        obj.pca = d["pca"]
        obj.feature_names_ = d["feature_names_"]
        return obj