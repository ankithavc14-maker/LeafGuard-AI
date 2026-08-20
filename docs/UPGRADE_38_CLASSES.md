# LeafGuard AI 3.0 — 38-class model upgrade

This version expands the disease classifier from the original 15 classes to 38 PlantVillage classes across 14 crop species.

## First run

The backend uses the Hugging Face model `Kathir56/plant-disease-tamilnadu`. On first startup, Transformers downloads the model and caches it locally. Subsequent runs reuse the cache.

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app:app
```

Open `http://127.0.0.1:8000`.

## Important scope

This is **not an all-plants classifier**. It covers 38 PlantVillage classes across 14 crop species: apple, blueberry, cherry, corn/maize, grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry and tomato.

The confidence threshold is a review safeguard, not true out-of-distribution detection. A future version should add a dedicated leaf/plant detector and OOD model.
