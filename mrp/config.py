"""Central configuration. Edit paths or parameters here, not in other modules."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Try loading with utf-8-sig (handles BOM), fallback to utf-16 if needed
try:
    load_dotenv(encoding='utf-8-sig')
except UnicodeDecodeError:
    try:
        load_dotenv(encoding='utf-16')
    except Exception:
        pass

# ── Paths ──────────────────────────────────────────────────────────────────
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
IMDB_DIR = DATA_DIR / "imdb"
RATINGS_CSV = DATA_DIR / "ratings.csv"
CACHE_DB = DATA_DIR / "cache.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── OMDb ───────────────────────────────────────────────────────────────────
OMDB_API_KEYS = [k.strip() for k in os.getenv("OMDB_API_KEYS", "").split(",") if k.strip()]
OMDB_URL = "https://www.omdbapi.com/"
OMDB_DAILY_LIMIT = 1000

# ── IMDb non-commercial datasets ───────────────────────────────────────────
IMDB_DATASETS = {
    "title.basics":     "https://datasets.imdbws.com/title.basics.tsv.gz",
    "title.ratings":    "https://datasets.imdbws.com/title.ratings.tsv.gz",
    "title.crew":       "https://datasets.imdbws.com/title.crew.tsv.gz",
    "title.principals": "https://datasets.imdbws.com/title.principals.tsv.gz",
    "name.basics":      "https://datasets.imdbws.com/name.basics.tsv.gz",
}

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
# PCA_COMPONENTS = 64
# PCA_COMPONENTS = 120 # +0.1602
# PCA_COMPONENTS = 128 # +0.1673
# PCA_COMPONENTS = 132 # +0.1569
# PCA_COMPONENTS = 140 # +0.1608

# PCA_COMPONENTS = 126 # +0.1696
PCA_COMPONENTS = 128 # +0.1701
# PCA_COMPONENTS = 130 # +0.1684

# ── Feature engineering ────────────────────────────────────────────────────
TOP_N_DIRECTORS = 100
TOP_N_ACTORS = 200

# ── Model ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS = 5
MIN_RATINGS_TO_TRAIN = 50

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,           # Reverted from 127
    "max_depth": -1,            # Remove max depth limit
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "min_data_in_leaf": 20,     # Reverted from 5
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "n_estimators": 500,
}


#   Mean MAE  = 1.0776  ± 0.0294
#   Mean RMSE = 1.3901  ± 0.0422
#   vs. baseline MAE 1.1764  →  improvement +0.0988