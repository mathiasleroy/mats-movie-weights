"""
Human-readable explanation of a single prediction.

Uses LightGBM SHAP contributions (pred_contrib=True), which are exactly
additive:  base_value + sum(contributions) == prediction.

Abstract features are grouped so the output is interpretable:
  emb__*      -> "Plot vibe"        (sum of all PCA embedding dims)
  genre__*    -> "Genres"
  dir__*      -> "Director"
  actor__*    -> "Cast"
  type__*     -> "Title type"
  everything else stays as its own factor
"""
import json
import numpy as np

from mrp.config import MODELS_DIR

# display labels for raw numeric features
_LABELS = {
    "imdb_rating": "IMDb rating",
    "log_votes": "Popularity (votes)",
    "runtime": "Runtime",
    "dir_te": "Director (your history)",
    "cast_te": "Cast (your history)",
    "cast_te_max": "Best actor (your history)",
    "cast_te_min": "Worst actor (your history)",
    "writer_te": "Writers (your history)",
    "writer_te_max": "Best writer (your history)",
    "writer_te_min": "Worst writer (your history)",
    "metascore": "Metascore",
    "log_box_office": "Box office",
    "log_awards_wins": "Award wins",
    "log_awards_noms": "Award nominations",
}

_GROUPS = [
    ("emb__", "Plot vibe"),
    ("genre__", "Genres"),
    ("type__", "Title type"),
    ("rated__", "Certificate"),
    ("country__", "Countries"),
    ("lang__", "Languages"),
]


def explain(model, builder, vec):
    """
    Parameters
    ----------
    model : lgb.Booster
    builder : FeatureBuilder (loaded, has feature_names_)
    vec : 1-D feature vector from builder.transform()

    Returns
    -------
    dict with keys:
      base            – model's base value (= mean prediction on training data)
      prediction      – base + sum of contributions
      factors         – list of {name, value, contribution}, sorted by |contribution|
      plot_available  – False if the movie had no plot text (zero embedding)
      mae             – CV MAE from metadata.json if available (else None)
    """
    shap = model.predict(vec.reshape(1, -1), pred_contrib=True)[0]
    base = float(shap[-1])
    contribs = shap[:-1]

    # group contributions by prefix
    groups = {label: 0.0 for _, label in _GROUPS}
    factors = []
    top_names = {}  # group label -> (name, contribution) with largest |contrib|

    for name, val in zip(builder.feature_names_, contribs):
        val = float(val)
        matched = False
        for prefix, label in _GROUPS:
            if name.startswith(prefix):
                groups[label] += val
                if label not in top_names or abs(val) > abs(top_names[label][1]):
                    top_names[label] = (name.split("__", 1)[1], val)
                matched = True
                break
        if not matched:
            raw_val = float(vec[list(builder.feature_names_).index(name)])
            factors.append({
                "name": _LABELS.get(name, name),
                "value": raw_val,
                "contribution": val,
            })

    # insert grouped factors (with a hint of the dominant member)
    # also collect top individual members per group for detailed output
    members = {label: [] for _, label in _GROUPS}
    for name, val in zip(builder.feature_names_, contribs):
        val = float(val)
        for prefix, label in _GROUPS:
            if name.startswith(prefix):
                members[label].append((name.split("__", 1)[1], val))
                break
    for label in members:
        members[label].sort(key=lambda t: abs(t[1]), reverse=True)

    for _, label in _GROUPS:
        total = groups[label]
        detail_name, detail_val = top_names.get(label, (None, 0.0))
        display = label
        if detail_name and abs(detail_val) > 0.01 and label != "Plot vibe":
            display = f"{label} ({detail_name})"
        factors.append({
            "name": display,
            "value": None,
            "contribution": total,
            "members": [
                {"name": n, "contribution": round(v, 4)}
                for n, v in members[label][:3] if abs(v) > 1e-6
            ],
        })

    factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)

    # was there any real plot text? (builder zero-fills missing embeddings,
    # so a zero embedding vector means no plot was available)
    idx = list(builder.feature_names_)
    first_emb = idx.index("emb__0")
    n_emb = sum(1 for n in idx if n.startswith("emb__"))
    plot_available = not np.allclose(vec[first_emb:first_emb + n_emb], 0.0)

    # confidence from training metadata
    mae = None
    meta_path = MODELS_DIR / "metadata.json"
    try:
        with open(meta_path) as f:
            mae = json.load(f).get("cv_mae_mean")
    except Exception:
        pass

    # sanitize: NaN/Inf (e.g. missing runtime) are not valid JSON
    def _clean_num(v):
        if v is None:
            return None
        v = float(v)
        return round(v, 4) if np.isfinite(v) else None

    for f in factors:
        f["value"] = _clean_num(f.get("value"))
        f["contribution"] = _clean_num(f["contribution"])

    return {
        "base": round(base, 3),
        "prediction": round(base + sum(float(c) for c in contribs), 3),
        "factors": factors,
        "plot_available": bool(plot_available),
        "mae": mae,
    }


def format_explanation(exp):
    """Plain-text rendering used by debug.py."""
    lines = []
    pred = exp["prediction"]
    rng = (
        f" (likely {pred - exp['mae']:.1f}–{pred + exp['mae']:.1f})"
        if exp.get("mae") else ""
    )
    lines.append(f"Base Rating (Your average): {exp['base']:.2f}")
    lines.append(f"Final Prediction: {pred:.2f}{rng}")
    if not exp["plot_available"]:
        lines.append("⚠ No plot text found — prediction based on metadata only.")
    lines.append("\n--- 24 FACTOR BREAKDOWN ---")
    for f in exp["factors"][:24]:
        val = f" ({f['value']:g})" if f.get("value") is not None else ""
        lines.append(f"  {f['contribution']:+.3f} pts  |  {f['name']}{val}")
        # show top individual members of grouped factors
        for m in f.get("members", [])[:3]:
            lines.append(f"      {m['contribution']:+.3f}     └ {m['name']}")
    return "\n".join(lines)
