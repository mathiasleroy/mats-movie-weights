"""OMDb API client — fetches plot text + structured metadata with rate-limit tracking."""
import requests
from mrp.config import OMDB_API_KEYS, OMDB_URL, OMDB_DAILY_LIMIT
from mrp.cache import get_cache


def _get_available_key():
    """Find the first API key that hasn't hit the daily limit."""
    cache = get_cache()
    for i, key in enumerate(OMDB_API_KEYS):
        if cache.get_omdb_count_today(i) < OMDB_DAILY_LIMIT:
            return i, key
    return None, None


def _parse_response(data):
    """Convert raw OMDb JSON into our feature dict format."""
    if not data or data.get("Response") == "False":
        return None
    return {
        "title": data.get("Title", ""),
        "title_type": _map_type(data.get("Type", "")),
        "year": _parse_year(data.get("Year")),
        "runtime": _parse_runtime(data.get("Runtime")),
        "genres": _split(data.get("Genre")),
        "directors": _split(data.get("Director")),
        "cast": _split(data.get("Actors")),
        "imdb_rating": _safe_float(data.get("imdbRating")),
        "imdb_votes": _safe_int(data.get("imdbVotes")),
        "plot": data.get("Plot", "") if data.get("Plot") not in ("", "N/A") else "",
    }


def fetch(imdb_id):
    if not OMDB_API_KEYS:
        return None

    cache = get_cache()
    key_idx, api_key = _get_available_key()
    if api_key is None:
        print(f"  ⚠ OMDb daily limit reached for ALL keys")
        return None

    cache.increment_omdb_count(key_idx, 1)

    try:
        resp = requests.get(
            OMDB_URL,
            params={"i": imdb_id, "apikey": api_key, "plot": "full"},
            timeout=15,
        )
    except requests.RequestException as exc:
        print(f"  OMDb request error for {imdb_id}: {exc}")
        return None

    if resp.status_code != 200:
        print(f"  OMDb HTTP {resp.status_code} for {imdb_id}")
        return None

    return _parse_response(resp.json())


def search_by_title(title, year=None):
    """Use OMDb's ?t= endpoint. Returns (imdb_id, parsed_data) to save API calls."""
    if not OMDB_API_KEYS:
        return None, None

    cache = get_cache()
    key_idx, api_key = _get_available_key()
    if api_key is None:
        return None, None
    cache.increment_omdb_count(key_idx, 1)

    params = {"t": title, "apikey": api_key, "plot": "full"}
    if year:
        params["y"] = year

    try:
        resp = requests.get(OMDB_URL, params=params, timeout=15)
        if resp.status_code != 200:
            return None, None
        data = resp.json()
        parsed = _parse_response(data)
        if parsed:
            imdb_id = data.get("imdbID", "")
            if imdb_id.startswith("tt"):
                return imdb_id, parsed
    except Exception:
        pass
    return None, None


# ── helpers ────────────────────────────────────────────────────────────────

def _map_type(t):
    return {"movie": "movie", "series": "tvSeries", "episode": "tvEpisode"}.get(t, t)


def _parse_year(s):
    if not s or s == "N/A":
        return None
    try:
        return int(str(s).split("–")[0].split("-")[0])
    except (ValueError, IndexError):
        return None


def _parse_runtime(s):
    if not s or s == "N/A":
        return None
    try:
        return int("".join(c for c in s if c.isdigit()))
    except ValueError:
        return None


def _split(s):
    if not s or s == "N/A":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _safe_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _safe_int(s):
    if not s or s == "N/A":
        return None
    try:
        return int(s.replace(",", ""))
    except (TypeError, ValueError):
        return None