"""OMDb API client — fetches plot text + structured metadata.

Uses a round-robin over all configured API keys: when a request fails
with an exhausted/invalid-key signal (HTTP 401 or "limit reached" in the
response body), the next key is tried automatically. No call counting.
"""
import requests
from mrp.config import OMDB_API_KEYS, OMDB_URL
from mrp.cache import get_cache

# Module-level cursor: index of the key currently in use.
_key_idx = None


def _current_key():
    """Return (index, api_key) for the active key, loading the persisted cursor once."""
    global _key_idx
    if _key_idx is None:
        cache = get_cache()
        stored = cache.get_setting("omdb_key_idx")
        _key_idx = int(stored) % len(OMDB_API_KEYS) if stored is not None else 0
    return _key_idx, OMDB_API_KEYS[_key_idx]


def _rotate_key():
    """Advance to the next key. Returns False if we wrapped around to the start."""
    global _key_idx
    start = _key_idx if _key_idx is not None else 0
    _key_idx = (_key_idx + 1) % len(OMDB_API_KEYS)
    get_cache().set_setting("omdb_key_idx", str(_key_idx))
    return _key_idx != start


def _is_limit_error(status_code, data):
    """Detect exhausted/invalid-key responses that warrant rotating keys."""
    if status_code == 401:
        return True
    err = (data or {}).get("Error", "")
    return "limit" in err.lower()


def _request(params):
    """Perform one OMDb request, rotating through keys on limit errors.

    Returns parsed JSON dict, or None if every key failed.
    """
    if not OMDB_API_KEYS:
        return None

    while True:
        _, api_key = _current_key()
        try:
            resp = requests.get(OMDB_URL, params={**params, "apikey": api_key},
                                timeout=15)
        except requests.RequestException as exc:
            print(f"  OMDb request error: {exc}")
            return None

        data = None
        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                pass

        if not _is_limit_error(resp.status_code, data):
            return data

        # Key exhausted/invalid -> silently switch to the next one.
        if not _rotate_key():
            print("  ⚠ OMDb daily limit reached for ALL keys")
            return None


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
        "writers": _split(data.get("Writer")),
        "imdb_rating": _safe_float(data.get("imdbRating")),
        "imdb_votes": _safe_int(data.get("imdbVotes")),
        "plot": data.get("Plot", "") if data.get("Plot") not in ("", "N/A") else "",
        # ── enrichment fields (same API call, no extra cost) ──
        "metascore": _safe_int(data.get("Metascore")),
        "rated": data.get("Rated") if data.get("Rated") not in ("", "N/A", None) else None,
        "languages": _split(data.get("Language")),
        "countries": _split(data.get("Country")),
        "awards_wins": _parse_awards(data.get("Awards"), "win"),
        "awards_nominations": _parse_awards(data.get("Awards"), "nomination"),
        "box_office": _parse_money(data.get("BoxOffice")),
    }


def fetch(imdb_id):
    data = _request({"i": imdb_id, "plot": "full"})
    return _parse_response(data)


def search_by_title(title, year=None):
    """Use OMDb's ?t= endpoint. Returns (imdb_id, parsed_data) to save API calls."""
    params = {"t": title, "plot": "full"}
    if year:
        params["y"] = year

    data = _request(params)
    parsed = _parse_response(data)
    if parsed:
        imdb_id = (data or {}).get("imdbID", "")
        if imdb_id.startswith("tt"):
            return imdb_id, parsed
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


def _parse_awards(s, kind):
    """Extract counts from strings like 'Nominated for 1 Oscar. 15 wins & 62 nominations total'."""
    if not s or s == "N/A":
        return None
    import re
    total = 0
    for m in re.finditer(r"(\d+)\s+" + kind, s, re.IGNORECASE):
        total += int(m.group(1))
    return total if total > 0 else None


def _parse_money(s):
    """'$461,172,890' -> 461172890"""
    if not s or s == "N/A":
        return None
    digits = "".join(c for c in s if c.isdigit())
    return int(digits) if digits else None
