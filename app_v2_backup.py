import io
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageStat, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "leafguard.db"
MODEL_PATH = BASE_DIR / "leafguard_model.pth"

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

DISPLAY = {
    "Pepper__bell___Bacterial_spot": ("Pepper", "Bacterial Spot"),
    "Pepper__bell___healthy": ("Pepper", "Healthy"),
    "Potato___Early_blight": ("Potato", "Early Blight"),
    "Potato___Late_blight": ("Potato", "Late Blight"),
    "Potato___healthy": ("Potato", "Healthy"),
    "Tomato_Bacterial_spot": ("Tomato", "Bacterial Spot"),
    "Tomato_Early_blight": ("Tomato", "Early Blight"),
    "Tomato_Late_blight": ("Tomato", "Late Blight"),
    "Tomato_Leaf_Mold": ("Tomato", "Leaf Mold"),
    "Tomato_Septoria_leaf_spot": ("Tomato", "Septoria Leaf Spot"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("Tomato", "Spider Mites"),
    "Tomato__Target_Spot": ("Tomato", "Target Spot"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("Tomato", "Yellow Leaf Curl Virus"),
    "Tomato__Tomato_mosaic_virus": ("Tomato", "Mosaic Virus"),
    "Tomato_healthy": ("Tomato", "Healthy"),
}

GUIDANCE = {
    "Healthy": {
        "about": "The model found no disease pattern among its supported classes.",
        "treatment": "Continue regular watering, nutrition, sanitation, and routine inspection.",
        "prevention": "Keep leaves dry when possible and monitor for new spots, discoloration, or pests."
    },
    "Early Blight": {
        "about": "A fungal disease that commonly causes dark lesions and can spread through older leaves.",
        "treatment": "Remove heavily affected leaves and follow locally approved fungicide guidance.",
        "prevention": "Improve airflow, avoid overhead watering, remove plant debris, and rotate crops."
    },
    "Late Blight": {
        "about": "A serious disease that can produce dark, water-soaked lesions and spread rapidly in humid conditions.",
        "treatment": "Isolate affected plants and follow local agricultural guidance for approved control measures.",
        "prevention": "Reduce leaf wetness, improve airflow, and remove infected material promptly."
    },
    "Bacterial Spot": {
        "about": "Bacterial infection associated with small dark spots on leaves and fruit.",
        "treatment": "Remove severely affected material and follow local extension guidance for management.",
        "prevention": "Use clean planting material, reduce splash, and sanitize tools."
    },
    "Leaf Mold": {
        "about": "A fungal disease that can appear as pale spots with mold growth, especially under humid conditions.",
        "treatment": "Remove affected foliage and improve ventilation; use locally approved controls when necessary.",
        "prevention": "Reduce humidity around foliage and avoid prolonged leaf wetness."
    },
    "Septoria Leaf Spot": {
        "about": "A fungal leaf-spot disease that often produces numerous small lesions.",
        "treatment": "Remove affected leaves and follow local disease-management guidance.",
        "prevention": "Use sanitation, crop rotation, airflow, and careful watering practices."
    },
    "Spider Mites": {
        "about": "Spider mites can cause stippling, yellowing, and general leaf decline.",
        "treatment": "Inspect leaf undersides and use an appropriate locally approved control if infestation is confirmed.",
        "prevention": "Monitor regularly and avoid plant stress and excessively dusty conditions."
    },
    "Target Spot": {
        "about": "A fungal leaf-spot disease that can resemble other tomato spot diseases.",
        "treatment": "Remove affected foliage and follow local extension guidance for approved fungicide use.",
        "prevention": "Improve airflow, reduce leaf wetness, and remove crop debris."
    },
    "Yellow Leaf Curl Virus": {
        "about": "A viral disease commonly associated with yellowing, curling, and stunted growth.",
        "treatment": "There is no direct cure for an infected plant; manage vectors and remove severely affected plants according to local guidance.",
        "prevention": "Monitor and manage whiteflies, use healthy planting material, and control volunteer plants."
    },
    "Mosaic Virus": {
        "about": "A viral disease that can produce mottled or mosaic-like leaf patterns.",
        "treatment": "Remove severely infected plants and follow local agricultural guidance.",
        "prevention": "Sanitize tools, use healthy planting material, and manage vectors where applicable."
    },
}

app = FastAPI(title="LeafGuard AI", version="2.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = models.mobilenet_v2(weights=None)
MODEL.classifier[1] = nn.Linear(MODEL.last_channel, len(CLASS_NAMES))
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
MODEL = MODEL.to(DEVICE)
MODEL.eval()

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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


def prettify(raw: str):
    plant, disease = DISPLAY[raw]
    return plant, disease


def guidance_for(disease: str):
    return GUIDANCE.get(disease, {
        "about": "The model detected a supported disease class.",
        "treatment": "Use local agricultural guidance before applying any treatment.",
        "prevention": "Inspect plants regularly and maintain good crop hygiene."
    })


def predict_tensor(tensor):
    with torch.no_grad():
        outputs = MODEL(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)
    return probs.cpu().numpy(), float(confidence.item()), int(pred_idx.item())


def save_prediction(record):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (record["id"], record["filename"], record["image_path"], record["plant"],
             record["disease"], record["confidence"], record["status"], record["created_at"])
        )
        conn.commit()


init_db()


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "MobileNetV2", "device": str(DEVICE), "classes": len(CLASS_NAMES)}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Please upload a JPG, JPEG, PNG, or WEBP image.")

    raw = await file.read()
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum image size is 5 MB.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded image could not be read.")

    ok, message = image_quality(image)
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    probs, confidence, pred_idx = predict_tensor(tensor)
    raw_label = CLASS_NAMES[pred_idx]
    plant, disease = prettify(raw_label)

    # Practical closed-set safeguard: low-confidence inputs are flagged instead of presented as certain diagnoses.
    if confidence < 0.55:
        status = "Needs review"
        disease_display = "Low-confidence result"
    else:
        status = "Prediction"
        disease_display = disease

    record_id = uuid.uuid4().hex[:12]
    safe_name = f"{record_id}_{Path(file.filename or 'leaf.jpg').name.replace(' ', '_')}"
    image_path = UPLOAD_DIR / safe_name
    image.save(image_path, format="JPEG", quality=90)
    created = datetime.now().isoformat(timespec="seconds")
    record = {
        "id": record_id, "filename": file.filename or safe_name, "image_path": str(image_path),
        "plant": plant, "disease": disease_display, "confidence": round(confidence * 100, 2),
        "status": status, "created_at": created
    }
    save_prediction(record)

    top3 = sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)[:3]
    top3_result = []
    for label, score in top3:
        p, d = prettify(label)
        top3_result.append({"plant": p, "disease": d, "confidence": round(float(score) * 100, 2)})

    g = guidance_for(disease)
    return {
        "id": record_id,
        "plant": plant,
        "disease": disease_display,
        "raw_prediction": disease,
        "confidence": round(confidence * 100, 2),
        "status": status,
        "about": g["about"],
        "treatment": g["treatment"],
        "prevention": g["prevention"],
        "top3": top3_result,
        "image_url": f"/api/image/{record_id}"
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
