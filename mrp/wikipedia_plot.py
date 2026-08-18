"""Fetch plot text from Wikipedia as a fallback / supplement to OMDb."""
import re
import time
import requests

_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "MovieRatingPredictor/1.0 (personal project; python-requests)"}
_PLOT_SECTION_NAMES = {"plot", "synopsis", "premise", "summary"}


def fetch_plot(title, year=None):
    """
    Return plot text for *title* from English Wikipedia, or ``None``.
    """
    search_query = f"{title} (film)" if year else title

    # Step 1: search
    try:
        resp = requests.get(
            _API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "srlimit": 5,
                "format": "json",
            },
            headers=_HEADERS,
            timeout=15,
        )
        results = resp.json().get("query", {}).get("search", [])
    except Exception:
        return None

    if not results:
        # retry without "(film)"
        if year:
            return fetch_plot(title, year=None)
        return None

    page_title = results[0]["title"]

    # Step 2: get section list and find plot section
    try:
        resp = requests.get(
            _API,
            params={
                "action": "parse",
                "page": page_title,
                "prop": "sections",
                "format": "json",
            },
            headers=_HEADERS,
            timeout=15,
        )
        sections = resp.json().get("parse", {}).get("sections", [])
    except Exception:
        sections = []

    plot_index = None
    for sec in sections:
        if sec.get("line", "").lower().strip() in _PLOT_SECTION_NAMES:
            plot_index = sec.get("index")
            break

    if plot_index is not None:
        # Step 3a: fetch the specific section
        text = _fetch_section(page_title, plot_index)
        if text and len(text) > 50:
            return text

    # Step 3b: fetch full wikitext and extract plot manually
    text = _fetch_full_and_extract(page_title)
    return text


def _fetch_section(page_title, section_index):
    try:
        resp = requests.get(
            _API,
            params={
                "action": "parse",
                "page": page_title,
                "prop": "wikitext",
                "section": section_index,
                "format": "json",
            },
            headers=_HEADERS,
            timeout=15,
        )
        wikitext = resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
        return _clean(wikitext)
    except Exception:
        return None


def _fetch_full_and_extract(page_title):
    try:
        resp = requests.get(
            _API,
            params={
                "action": "parse",
                "page": page_title,
                "prop": "wikitext",
                "format": "json",
            },
            headers=_HEADERS,
            timeout=15,
        )
        wikitext = resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
    except Exception:
        return None

    m = re.search(
        r"(?:^|\n)==\s*(?:Plot|Synopsis|Premise|Summary)\s*==(.*?)(?:\n==\s|\Z)",
        wikitext,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return _clean(m.group(1))
    return None


def _clean(text):
    """Strip wikitext markup → plain text."""
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"={2,}[^=]+={2,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()