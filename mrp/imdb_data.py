"""Download, load, and query the IMDb non-commercial datasets."""
import pandas as pd
import requests
from tqdm import tqdm
from mrp.config import IMDB_DIR, IMDB_DATASETS


class IMDbData:
    """Holds the five core IMDb TSV tables in memory as DataFrames."""

    def __init__(self):
        self.loaded = False
        self.basics = None
        self.ratings = None
        self.crew = None
        self.principals = None
        self.names = None

    # ── Download ───────────────────────────────────────────────────────────

    def download(self):
        """Download any missing IMDb TSV gzips."""
        IMDB_DIR.mkdir(parents=True, exist_ok=True)
        for name, url in IMDB_DATASETS.items():
            filepath = IMDB_DIR / f"{name}.tsv.gz"
            if filepath.exists():
                print(f"  ✓ {name} (already present)")
                continue
            print(f"  ↓ Downloading {name} …")
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(filepath, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=name
            ) as bar:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            print(f"  ✓ {name} done")

    # ── Load ───────────────────────────────────────────────────────────────

    def load(self, rated_tconsts=None):
        """
        Load all datasets into memory.

        Parameters
        ----------
        rated_tconsts : set or None
            If provided, ``title.principals`` is filtered to only these IDs
            (the file is ~3 GB uncompressed otherwise).
        """
        self._load_basics()
        self._load_ratings()
        self._load_crew()
        self._load_names()
        self._load_principals(rated_tconsts)
        self.loaded = True
        print(
            f"  IMDb data loaded — "
            f"basics={len(self.basics)}, ratings={len(self.ratings)}, "
            f"crew={len(self.crew)}, principals={len(self.principals)}, "
            f"names={len(self.names)}"
        )

    def _load_basics(self):
        # Read as strings first to avoid crashes on malformed TSV rows
        df = pd.read_csv(
            IMDB_DIR / "title.basics.tsv.gz",
            sep="\t",
            na_values="\\N",
            dtype=str,
            usecols=[
                "tconst", "titleType", "primaryTitle", "originalTitle",
                "isAdult", "startYear", "endYear", "runtimeMinutes", "genres",
            ],
            low_memory=False,
        )
        # Safely convert to numbers
        df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce").astype("Int64")
        df["endYear"] = pd.to_numeric(df["endYear"], errors="coerce").astype("Int64")
        df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce").astype("Int64")
        self.basics = df.set_index("tconst")

    def _load_ratings(self):
        df = pd.read_csv(
            IMDB_DIR / "title.ratings.tsv.gz",
            sep="\t",
            na_values="\\N",
            dtype=str,
        )
        df["averageRating"] = pd.to_numeric(df["averageRating"], errors="coerce")
        df["numVotes"] = pd.to_numeric(df["numVotes"], errors="coerce").astype("Int64")
        self.ratings = df.set_index("tconst")

    def _load_crew(self):
        self.crew = pd.read_csv(
            IMDB_DIR / "title.crew.tsv.gz",
            sep="\t",
            na_values="\\N",
            dtype={"tconst": str, "directors": str, "writers": str},
        ).set_index("tconst")

    def _load_names(self):
        self.names = pd.read_csv(
            IMDB_DIR / "name.basics.tsv.gz",
            sep="\t",
            na_values="\\N",
            usecols=["nconst", "primaryName"],
            dtype={"nconst": str, "primaryName": str},
        ).set_index("nconst")

    def _load_principals(self, rated_tconsts=None):
        cols = ["tconst", "ordering", "nconst", "category"]
        if rated_tconsts is not None:
            rated_set = set(rated_tconsts)
            chunks = []
            for chunk in pd.read_csv(
                IMDB_DIR / "title.principals.tsv.gz",
                sep="\t",
                na_values="\\N",
                chunksize=500_000,
                usecols=cols,
                dtype=str,
            ):
                mask = chunk["tconst"].isin(rated_set)
                if mask.any():
                    chunks.append(chunk[mask])
            self.principals = (
                pd.concat(chunks, ignore_index=True)
                if chunks
                else pd.DataFrame(columns=cols)
            )
        else:
            self.principals = pd.read_csv(
                IMDB_DIR / "title.principals.tsv.gz",
                sep="\t",
                na_values="\\N",
                usecols=cols,
                dtype=str,
            )
            
        # Safely convert ordering
        self.principals["ordering"] = pd.to_numeric(self.principals["ordering"], errors="coerce").astype("Int64")

    # ── Lookup ─────────────────────────────────────────────────────────────

    def lookup(self, tconst):
        """Return a feature dict for *tconst*, or ``None`` if not found."""
        if not self.loaded or tconst not in self.basics.index:
            return None

        b = self.basics.loc[tconst]
        result = {
            "title": _safe_str(b.get("primaryTitle")),
            "original_title": _safe_str(b.get("originalTitle")),
            "title_type": _safe_str(b.get("titleType")),
            "year": int(b["startYear"]) if pd.notna(b.get("startYear")) else None,
            "runtime": int(b["runtimeMinutes"]) if pd.notna(b.get("runtimeMinutes")) else None,
            "genres": _split_csv(b.get("genres")),
        }

        # public rating + votes
        if tconst in self.ratings.index:
            r = self.ratings.loc[tconst]
            result["imdb_rating"] = float(r["averageRating"]) if pd.notna(r.get("averageRating")) else None
            result["imdb_votes"] = int(r["numVotes"]) if pd.notna(r.get("numVotes")) else None
        else:
            result["imdb_rating"] = None
            result["imdb_votes"] = None

        # directors (from title.crew → resolve nconsts to names)
        result["directors"] = []
        if tconst in self.crew.index:
            d_str = self.crew.loc[tconst].get("directors")
            if pd.notna(d_str) and d_str:
                for nid in d_str.split(","):
                    name = self._resolve_name(nid)
                    if name:
                        result["directors"].append(name)

        # top cast (from title.principals)
        result["cast"] = []
        if self.principals is not None and len(self.principals) > 0:
            cast_rows = self.principals[
                (self.principals["tconst"] == tconst)
                & (self.principals["category"].isin(["actor", "actress"]))
            ].nsmallest(10, "ordering")
            for nid in cast_rows["nconst"]:
                name = self._resolve_name(nid)
                if name:
                    result["cast"].append(name)

        return result

    def _resolve_name(self, nconst):
        if nconst in self.names.index:
            val = self.names.loc[nconst, "primaryName"]
            return val if isinstance(val, str) else None
        return None

    # ── Title search ───────────────────────────────────────────────────────

    def search_title(self, title, year=None):
        """
        Find the best-matching tconst for *title*.
        Returns ``None`` if no match.
        """
        if not self.loaded:
            return None

        title_lower = title.lower().strip()

        # 1. exact primaryTitle match
        mask = self.basics["primaryTitle"].str.lower() == title_lower
        matches = self.basics[mask]
        if len(matches) == 0:
            mask = self.basics["originalTitle"].str.lower() == title_lower
            matches = self.basics[mask]
        if len(matches) == 0:
            mask = self.basics["primaryTitle"].str.contains(
                title_lower, case=False, na=False, regex=False
            )
            matches = self.basics[mask]
        if len(matches) == 0:
            return None

        # prefer non-adult, title types that make sense
        type_priority = {"movie": 0, "tvSeries": 1, "tvMiniSeries": 2,
                         "tvMovie": 3, "video": 4, "short": 5}
        matches = matches.copy()
        matches["_prio"] = matches["titleType"].map(type_priority).fillna(99)

        # filter by year if given
        if year is not None:
            year_matches = matches[matches["startYear"] == year]
            if len(year_matches) > 0:
                matches = year_matches

        # join vote counts and sort by notability
        matches = matches.join(
            self.ratings[["numVotes"]], how="left"
        ).sort_values(
            ["_prio", "numVotes"], ascending=[True, False], na_position="last"
        )
        return matches.index[0]


def _safe_str(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val)


def _split_csv(val):
    if val is None or (isinstance(val, float) and pd.isna(val)) or not val:
        return []
    return [g.strip() for g in str(val).split(",") if g.strip()]


# ── Singleton ──────────────────────────────────────────────────────────────
_instance = None


def get_instance() -> IMDbData:
    global _instance
    if _instance is None:
        _instance = IMDbData()
    return _instance