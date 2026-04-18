# S&P 500 Competitor Similarity API

This project serves precomputed company similarity using a lightweight FastAPI service.

The API blends two signals:
- TechOverlap: cosine similarity in text-derived feature space (`tech_embeddings.npy`)
- MarketBehavior: cosine similarity in learned behavior space (`behavior_embeddings.npy`)

Combined score:

`CombinedScore = w_tech * TechOverlap + (1 - w_tech) * MarketBehavior`

## Repository Files To Keep

- `similarity_api.py`
- `tech_embeddings.npy`
- `behavior_embeddings.npy`
- `ticker_lookup.csv`
- `Project.ipynb` (optional, for reproducibility)

## Quick Start

1. Create/activate environment (optional)
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run API:

```powershell
uvicorn similarity_api:app --host 127.0.0.1 --port 8000
```

## Endpoints

### Health

`GET /health`

Example:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/health"
```

### Competitors for one weight

`GET /competitors/{ticker}?k=5&w_tech=0.6&same_sector_only=false`

Example:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/competitors/UNH?k=5&w_tech=0.6"
```

### Weight sweep

`GET /competitors/{ticker}/sweep?k=5&weights_csv=0.0,0.6,1.0`

Example:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/competitors/MSFT/sweep?k=3&weights_csv=0.0,0.6,1.0"
```

## Notes On Git Ignore Strategy

This repo currently ignores training artifacts (`sp500_*`, `.pt`, `.onnx`, and scratch outputs) to keep Git history clean and small.

If you want to version full reproducibility artifacts later, remove specific ignore lines and commit those files explicitly.

## Suggested First Commit

```powershell
git init
git add similarity_api.py tech_embeddings.npy behavior_embeddings.npy ticker_lookup.csv Project.ipynb requirements.txt README.md .gitignore
git commit -m "Initial competitor similarity API with precomputed embeddings"
```
