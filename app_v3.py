import io
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "leafguard.db"

# 38-class PlantVillage MobileNetV2 model. The model is downloaded and cached
# automatically by Hugging Face Transformers on first startup.
MODEL_ID = os.getenv("LEAFGUARD_MODEL", "Kathir56/plant-disease-tamilnadu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="LeafGuard AI", version="3.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

MODEL = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(DEVICE)
PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_ID)
MODEL.eval()

ID2LABEL = {int(k): v for k, v in MODEL.config.id2label.items()}
CLASS_NAMES = [ID2LABEL[i] for i in sorted(ID2LABEL)]

SUPPORTED_PLANTS = sorted({label.split("___")[0].replace("_", " ") for label in CLASS_NAMES})

DISPLAY_PLANTS = {
    "Apple": "Apple", "Blueberry": "Blueberry", "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Corn / Maize", "Grape": "Grape", "Orange": "Orange / Citrus",
    "Peach": "Peach", "Pepper,_bell": "Bell Pepper", "Potato": "Potato",
    "Raspberry": "Raspberry", "Soybean": "Soybean", "Squash": "Squash",
    "Strawberry": "Strawberry", "Tomato": "Tomato",
}

GUIDANCE = {
    "Healthy": {
        "about": "No disease pattern was detected among the supported classes.",
        "treatment": "No disease treatment is indicated from this prediction. Continue normal crop care and monitoring.",
        "prevention": "Maintain good sanitation, appropriate watering, airflow, nutrition, and regular inspection."
    },
    "Apple scab": {"about": "A fungal disease that causes olive or dark lesions on apple leaves and fruit.", "treatment": "Remove affected material and follow local extension guidance for approved fungicide programs.", "prevention": "Improve sanitation, remove fallen leaves, and maintain airflow."},
    "Black rot": {"about": "A fungal disease associated with dark lesions and fruit rot in susceptible crops.", "treatment": "Remove infected material and follow locally approved disease-management guidance.", "prevention": "Sanitize plant debris, improve airflow, and avoid prolonged leaf wetness."},
    "Cedar apple rust": {"about": "A fungal disease affecting apple leaves, often producing yellow-orange lesions.", "treatment": "Follow local extension guidance for management and approved fungicides.", "prevention": "Monitor early symptoms and manage nearby alternate hosts where appropriate."},
    "Powdery mildew": {"about": "A fungal disease that produces a characteristic white powdery growth on foliage.", "treatment": "Remove severely affected material and follow local agricultural guidance for approved controls.", "prevention": "Improve airflow and avoid conditions that keep foliage crowded and humid."},
    "Common rust": {"about": "A fungal rust disease that produces rust-colored pustules on crop leaves.", "treatment": "Use locally approved disease-management practices and resistant varieties where available.", "prevention": "Monitor crops regularly and maintain good field hygiene."},
    "Northern Leaf Blight": {"about": "A fungal disease of maize that can produce elongated leaf lesions.", "treatment": "Follow local extension recommendations for resistant varieties and approved controls.", "prevention": "Use resistant cultivars, crop rotation, and residue management where appropriate."},
    "Esca (Black Measles)": {"about": "A grapevine disease complex associated with leaf symptoms and decline.", "treatment": "Consult local viticulture guidance for confirmed cases and affected vines.", "prevention": "Use healthy planting material and good vineyard sanitation."},
    "Leaf blight (Isariopsis Leaf Spot)": {"about": "A fungal leaf disease affecting grape foliage.", "treatment": "Remove affected material and follow local extension guidance.", "prevention": "Improve canopy airflow and reduce prolonged leaf wetness."},
    "Haunglongbing (Citrus greening)": {"about": "A serious citrus disease associated with leaf mottling and plant decline.", "treatment": "Seek local plant-health guidance; management focuses on infected material and vector control.", "prevention": "Use certified planting material and monitor/control disease vectors according to local guidance."},
    "Bacterial spot": {"about": "A bacterial disease that can cause small dark spots on leaves and fruit.", "treatment": "Remove severely affected material and follow local extension recommendations.", "prevention": "Reduce splash, sanitize tools, and use healthy planting material."},
    "Early blight": {"about": "A disease that commonly produces dark lesions and can spread through older leaves.", "treatment": "Remove heavily affected leaves and follow locally approved fungicide guidance.", "prevention": "Improve airflow, avoid overhead watering, remove debris, and rotate crops."},
    "Late blight": {"about": "A serious disease that can produce dark, water-soaked lesions and spread rapidly under favorable conditions.", "treatment": "Isolate affected plants and follow local agricultural guidance for approved control measures.", "prevention": "Reduce leaf wetness, improve airflow, and remove infected material promptly."},
    "Leaf Mold": {"about": "A fungal disease that can appear as pale leaf spots with mold growth under humid conditions.", "treatment": "Remove affected foliage and improve ventilation; use locally approved controls when necessary.", "prevention": "Reduce humidity around foliage and avoid prolonged leaf wetness."},
    "Septoria leaf spot": {"about": "A fungal leaf-spot disease that often produces numerous small lesions.", "treatment": "Remove affected leaves and follow local disease-management guidance.", "prevention": "Use sanitation, crop rotation, airflow, and careful watering practices."},
    "Spider mites Two-spotted spider mite": {"about": "Spider mites can cause stippling, yellowing, and general leaf decline.", "treatment": "Inspect leaf undersides and use an appropriate locally approved control if infestation is confirmed.", "prevention": "Monitor regularly and avoid plant stress and excessively dusty conditions."},
    "Target Spot": {"about": "A fungal leaf-spot disease that can resemble other crop spot diseases.", "treatment": "Remove affected foliage and follow local extension guidance for approved fungicide use.", "prevention": "Improve airflow, reduce leaf wetness, and remove crop debris."},
    "Tomato Yellow Leaf Curl Virus": {"about": "A viral disease commonly associated with yellowing, curling, and stunted tomato growth.", "treatment": "There is no direct cure for an infected plant; manage vectors and follow local agricultural guidance.", "prevention": "Monitor and manage whiteflies, use healthy planting material, and control volunteer plants."},
    "Tomato mosaic virus": {"about": "A viral disease that can produce mottled or mosaic-like leaf patterns.", "treatment": "Remove severely infected plants and follow local agricultural guidance.", "prevention": "Sanitize tools, use healthy planting material, and manage vectors where applicable."},
}


def normalize_text(value: str) -> str:
    return value.replace("_", " ").replace("  ", " ").strip()


def parse_label(raw: str):
    if "___" in raw:
        plant_raw, disease_raw = raw.split("___", 1)
    else:
        plant_raw, disease_raw = "Unknown", raw
    plant = DISPLAY_PLANTS.get(plant_raw, normalize_text(plant_raw))
    disease = normalize_text(disease_raw)
    disease = disease.replace("Spider mites Two-spotted spider mite", "Spider Mites")
    disease = disease.replace("Tomato Yellow Leaf Curl Virus", "Yellow Leaf Curl Virus")
    disease = disease.replace("Tomato mosaic virus", "Mosaic Virus")
    disease = disease.replace("Leaf Mold", "Leaf Mold")
    return plant, disease


def guidance_for(disease: str):
    return GUIDANCE.get(disease, {
        "about": "The model detected a supported plant-health class.",
        "treatment": "Use local agricultural guidance before applying any treatment.",
        "prevention": "Inspect plants regularly and maintain good crop hygiene."
    })


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                image_path TEXT,
                plant TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def image_quality(image: Image.Image):
    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    variance = float(gray.var())
    brightness = float(gray.mean())
    if variance < 90:
        return False, "Image appears too blurry or low-detail. Please upload a clearer leaf image."
    if brightness < 25 or brightness > 245:
        return False, "Image lighting is too dark or too bright. Please upload a well-lit leaf image."
    return True, "OK"


def predict(image: Image.Image):
    inputs = PROCESSOR(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    top = torch.topk(probs, k=min(3, len(CLASS_NAMES)))
    results = []
    for idx, score in zip(top.indices.tolist(), top.values.tolist()):
        raw = ID2LABEL[int(idx)]
        plant, disease = parse_label(raw)
        results.append({"raw": raw, "plant": plant, "disease": disease, "confidence": round(float(score) * 100, 2)})
    return results


init_db()


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "architecture": MODEL.__class__.__name__,
        "device": str(DEVICE),
        "classes": len(CLASS_NAMES),
        "plants": len(SUPPORTED_PLANTS),
        "supported_plants": [DISPLAY_PLANTS.get(p, normalize_text(p)) for p in SUPPORTED_PLANTS],
    }


@app.get("/api/supported")
def supported():
    grouped = {}
    for raw in CLASS_NAMES:
        plant, disease = parse_label(raw)
        grouped.setdefault(plant, []).append(disease)
    return {"classes": len(CLASS_NAMES), "plants": grouped}


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Please upload a JPG, JPEG, PNG, or WEBP image.")
    raw_bytes = await file.read()
    if len(raw_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum image size is 5 MB.")
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded image could not be read.")

    ok, message = image_quality(image)
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    top3 = predict(image)
    best = top3[0]
    confidence = best["confidence"]
    # This is a review threshold, not a true OOD detector.
    status = "Prediction" if confidence >= 55 else "Needs review"
    disease_display = best["disease"] if confidence >= 55 else "Low-confidence result"
    g = guidance_for(best["disease"])

    record_id = uuid.uuid4().hex[:12]
    safe_name = f"{record_id}_{Path(file.filename or 'leaf.jpg').name.replace(' ', '_')}"
    image_path = UPLOAD_DIR / safe_name
    image.save(image_path, format="JPEG", quality=90)
    created = datetime.now().isoformat(timespec="seconds")
    record = {
        "id": record_id, "filename": file.filename or safe_name, "image_path": str(image_path),
        "plant": best["plant"], "disease": disease_display, "confidence": confidence,
        "status": status, "created_at": created
    }
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(record.values()))
        conn.commit()

    return {
        "id": record_id,
        "plant": best["plant"],
        "disease": disease_display,
        "raw_prediction": best["disease"],
        "confidence": confidence,
        "status": status,
        "about": g["about"],
        "treatment": g["treatment"],
        "prevention": g["prevention"],
        "top3": [{k: x[k] for k in ("plant", "disease", "confidence")} for x in top3],
        "model_classes": len(CLASS_NAMES),
        "image_url": f"/api/image/{record_id}",
    }


@app.get("/api/history")
def history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 30").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/stats")
def stats():
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        healthy = conn.execute("SELECT COUNT(*) FROM predictions WHERE disease='Healthy'").fetchone()[0]
        avg = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
    return {"total": total, "healthy": healthy, "diseased": total - healthy, "average_confidence": round(avg or 0, 2)}


@app.get("/api/image/{prediction_id}")
def get_image(prediction_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT image_path FROM predictions WHERE id=?", (prediction_id,)).fetchone()
    if not row or not Path(row[0]).exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(row[0])
