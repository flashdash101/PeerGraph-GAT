from __future__ import annotations

import csv
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query


app = FastAPI(title="S&P500 Competitor Similarity API", version="1.0.0")


# Load precomputed artifacts at startup.
tickers: list[str] = []
sectors: list[str] = []
with open("ticker_lookup.csv", "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tickers.append(str(row["Ticker"]).upper())
        sectors.append(str(row["GICS Sector"]))

tech_embeddings = np.load("tech_embeddings.npy").astype(np.float32)
behavior_embeddings = np.load("behavior_embeddings.npy").astype(np.float32)

if len(tickers) != tech_embeddings.shape[0] or len(tickers) != behavior_embeddings.shape[0]:
    raise RuntimeError("Artifact row counts do not match between lookup and embeddings.")

# Pre-normalize once, so each request is fast dot products only.
def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(denom, eps)


tech_norm = l2_normalize(tech_embeddings)
behavior_norm = l2_normalize(behavior_embeddings)

ticker_to_idx = {t: i for i, t in enumerate(tickers)}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "num_companies": int(len(tickers)),
        "tech_dim": int(tech_embeddings.shape[1]),
        "behavior_dim": int(behavior_embeddings.shape[1]),
    }


@app.get("/competitors/{ticker}")
def competitors(
    ticker: str,
    k: int = Query(5, ge=1, le=50),
    w_tech: float = Query(0.6, ge=0.0, le=1.0),
    same_sector_only: bool = Query(False),
    min_tech: Optional[float] = Query(None),
    min_behavior: Optional[float] = Query(None),
) -> dict:
    return _competitors_impl(
        ticker=ticker,
        k=k,
        w_tech=w_tech,
        same_sector_only=same_sector_only,
        min_tech=min_tech,
        min_behavior=min_behavior,
    )


def _competitors_impl(
    ticker: str,
    k: int = 5,
    w_tech: float = 0.6,
    same_sector_only: bool = False,
    min_tech: Optional[float] = None,
    min_behavior: Optional[float] = None,
) -> dict:
    t = ticker.upper().strip()
    if t not in ticker_to_idx:
        raise HTTPException(status_code=404, detail=f"Ticker not found: {t}")

    idx = ticker_to_idx[t]
    w_behavior = 1.0 - w_tech

    # Cosine similarity is dot product since vectors are pre-normalized.
    tech_sim = tech_norm @ tech_norm[idx]
    behavior_sim = behavior_norm @ behavior_norm[idx]
    combined = w_tech * tech_sim + w_behavior * behavior_sim

    # Remove self-match.
    mask = np.ones(len(tickers), dtype=bool)
    mask[idx] = False

    # Optional retrieval constraints for cleaner competitor sets.
    if same_sector_only:
        anchor_sector = sectors[idx]
        sector_mask = np.array([s == anchor_sector for s in sectors], dtype=bool)
        mask &= sector_mask
    if min_tech is not None:
        mask &= tech_sim >= float(min_tech)
    if min_behavior is not None:
        mask &= behavior_sim >= float(min_behavior)

    candidate_idx = np.where(mask)[0]
    if candidate_idx.size == 0:
        return {
            "ticker": t,
            "weights": {"tech": w_tech, "behavior": w_behavior},
            "count": 0,
            "results": [],
        }

    candidate_scores = combined[candidate_idx]
    top_n = min(k, candidate_idx.size)

    # Use argpartition for speed, then final sort on the small top-n set.
    partial = np.argpartition(-candidate_scores, top_n - 1)[:top_n]
    top_idx = candidate_idx[partial]
    top_idx = top_idx[np.argsort(-combined[top_idx])]

    results = []
    for i in top_idx.tolist():
        results.append(
            {
                "Ticker": tickers[i],
                "GICS Sector": sectors[i],
                "TechOverlap": float(tech_sim[i]),
                "MarketBehavior": float(behavior_sim[i]),
                "CombinedScore": float(combined[i]),
            }
        )

    return {
        "ticker": t,
        "weights": {"tech": w_tech, "behavior": w_behavior},
        "count": int(len(results)),
        "results": results,
    }


@app.get("/competitors/{ticker}/sweep")
def competitors_sweep(
    ticker: str,
    k: int = Query(5, ge=1, le=50),
    weights_csv: Optional[str] = Query(None),
    same_sector_only: bool = Query(False),
    min_tech: Optional[float] = Query(None),
    min_behavior: Optional[float] = Query(None),
) -> dict:
    return _competitors_sweep_impl(
        ticker=ticker,
        k=k,
        weights_csv=weights_csv,
        same_sector_only=same_sector_only,
        min_tech=min_tech,
        min_behavior=min_behavior,
    )


def _parse_weight_grid(weights_csv: Optional[str]) -> list[float]:
    if not weights_csv:
        return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    values = []
    for raw in weights_csv.split(","):
        s = raw.strip()
        if not s:
            continue
        w = float(s)
        if w < 0.0 or w > 1.0:
            raise HTTPException(status_code=400, detail=f"Weight out of range [0,1]: {w}")
        values.append(w)

    if not values:
        raise HTTPException(status_code=400, detail="No valid weights provided in weights_csv")

    return sorted(set(values))


def _competitors_sweep_impl(
    ticker: str,
    k: int = 5,
    weights_csv: Optional[str] = None,
    same_sector_only: bool = False,
    min_tech: Optional[float] = None,
    min_behavior: Optional[float] = None,
) -> dict:
    grid = _parse_weight_grid(weights_csv)

    sweeps = []
    for w_tech in grid:
        run = _competitors_impl(
            ticker=ticker,
            k=k,
            w_tech=float(w_tech),
            same_sector_only=same_sector_only,
            min_tech=min_tech,
            min_behavior=min_behavior,
        )
        sweeps.append(
            {
                "weights": run["weights"],
                "count": run["count"],
                "top": run["results"],
            }
        )

    return {
        "ticker": ticker.upper().strip(),
        "k": int(k),
        "num_weight_settings": int(len(sweeps)),
        "sweeps": sweeps,
    }


# Run locally:
# uvicorn similarity_api:app --host 0.0.0.0 --port 8000 --reload
