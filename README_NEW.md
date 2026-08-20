# 🌿 LeafGuard AI — Live AI Plant Disease Detection

An end-to-end upgrade of the original LeafGuard AI MobileNetV2 classifier.

## What was added
- Responsive frontend dashboard served by FastAPI
- Drag-and-drop image upload
- File type and 5 MB size validation
- Basic image-quality screening
- Low-confidence prediction safeguard
- Disease/plant display mapping
- Treatment and prevention guidance
- Top-3 alternative predictions
- SQLite prediction history
- Dashboard statistics
- Plant guide and project explanation pages
- Mobile-responsive design

## Original ML model
- MobileNetV2
- 15 classes
- Pepper, Potato and Tomato
- Original reported test accuracy: 94%

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open: `http://127.0.0.1:8000`

## Live demo deployment
Recommended simple options: Render or Railway for the FastAPI app. The frontend is already served by FastAPI, so there is no separate Node server required.

### Demo flow
1. Open the public URL.
2. Click **Predict Disease**.
3. Upload a clear tomato/potato/pepper leaf image.
4. Show disease + confidence + treatment/prevention.
5. Open **History** to demonstrate persistence.
6. Show **Plant Guide** and **About Project**.
7. Demonstrate an invalid/poor-quality image to explain validation.

## Important limitation
The low-confidence safeguard is a practical closed-set warning, not a true OOD detector. For a research-grade version, add a dedicated leaf/non-leaf model or calibrated OOD detection and evaluate it on external real-world datasets.
