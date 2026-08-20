# LeafGuard66

Final Phase 1-3 build with: 38-class plant-health model, semantic leaf/non-leaf gate, single upload workflow, Grad-CAM, authentication, history, statistics, plant guide, and responsive UI.

## Run

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn app:app
```

Open http://127.0.0.1:8000

For production, set a strong `SECRET_KEY`, use HTTPS, a persistent database, rate limiting, and validate agricultural guidance with qualified local experts.
