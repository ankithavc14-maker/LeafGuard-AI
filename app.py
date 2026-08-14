import base64
import hashlib
import io
import os
import secrets
import uuid
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import psycopg2
from psycopg2 import IntegrityError
from psycopg2.extras import RealDictCursor

load_dotenv()

from transformers import AutoImageProcessor, AutoModelForImageClassification, CLIPModel, CLIPProcessor
import jwt

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required. Render supplies it from the Postgres database in render.yaml; locally set it in .env.")

MODEL_ID = os.getenv("LEAFGUARD_MODEL", "Kathir56/plant-disease-tamilnadu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
REVIEW_THRESHOLD = 55.0
MARGIN_THRESHOLD = 8.0
APP_ENV = os.getenv("APP_ENV", "development").lower()
JWT_SECRET = os.getenv("SECRET_KEY") or os.getenv("LEAFGUARD_JWT_SECRET", "leafguard-demo-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_HOURS = 12
ENABLE_GRADCAM = os.getenv("LEAFGUARD_GRADCAM", "true").lower() != "false"

if APP_ENV == "production" and (JWT_SECRET == "leafguard-demo-secret-change-in-production" or len(JWT_SECRET) < 32):
    raise RuntimeError("SECRET_KEY must be set to a random value of at least 32 characters in production.")

app = FastAPI(title="LeafGuard AI", version="5.0")

# Explicitly serve JavaScript with an executable MIME type.
# Some Windows Python installations can classify .js as text/plain.
@app.get("/static/app.js", include_in_schema=False)
def serve_app_js():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Production hardening: security headers and lightweight per-process rate limiting.
# For multiple replicas, replace this limiter with a shared Redis-backed limiter.
class SecurityAndRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, predict_limit=30, auth_limit=10, window_seconds=60):
        super().__init__(app)
        self.predict_limit = predict_limit
        self.auth_limit = auth_limit
        self.window = window_seconds
        self.hits = defaultdict(deque)

    async def dispatch(self, request, call_next):
        client = request.client.host if request.client else "unknown"
        path = request.url.path
        limit = self.predict_limit if path == "/api/predict" else self.auth_limit if path in {"/api/auth/login", "/api/auth/register"} else None
        if limit is not None:
            now = time.monotonic()
            key = f"{client}:{path}"
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= limit:
                return JSONResponse({"detail": "Too many requests. Please try again later."}, status_code=429, headers={"Retry-After": str(self.window)})
            q.append(now)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if APP_ENV == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityAndRateLimitMiddleware)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
MODEL = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(DEVICE)
PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_ID)
MODEL.eval()
ID2LABEL = {int(k): v for k, v in MODEL.config.id2label.items()}
CLASS_NAMES = [ID2LABEL[i] for i in sorted(ID2LABEL)]

DISPLAY_PLANTS = {
    "Apple": "Apple", "Blueberry": "Blueberry", "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Corn / Maize", "Grape": "Grape", "Orange": "Orange / Citrus",
    "Peach": "Peach", "Pepper,_bell": "Bell Pepper", "Potato": "Potato",
    "Raspberry": "Raspberry", "Soybean": "Soybean", "Squash": "Squash",
    "Strawberry": "Strawberry", "Tomato": "Tomato",
}

SUPPORTED_PLANTS = sorted({label.split("___")[0] for label in CLASS_NAMES})

# -----------------------------------------------------------------------------
# Guidance
# -----------------------------------------------------------------------------
GUIDANCE = {
    "Healthy": {
        "about": "No disease pattern was detected among the supported classes.",
        "treatment": "No disease treatment is indicated from this prediction. Continue normal crop care and monitoring.",
        "prevention": "Maintain good sanitation, appropriate watering, airflow, nutrition, and regular inspection.",
    },
    "Apple scab": {"about": "A fungal disease that causes olive or dark lesions on apple leaves and fruit.", "treatment": "Remove affected material and follow local extension guidance for approved disease-management programs.", "prevention": "Improve sanitation, remove fallen leaves, and maintain airflow."},
    "Black rot": {"about": "A fungal disease associated with dark lesions and fruit rot in susceptible crops.", "treatment": "Remove infected material and follow locally approved disease-management guidance.", "prevention": "Sanitize plant debris, improve airflow, and avoid prolonged leaf wetness."},
    "Cedar apple rust": {"about": "A fungal disease affecting apple leaves, often producing yellow-orange lesions.", "treatment": "Follow local extension guidance for management and approved controls.", "prevention": "Monitor early symptoms and manage nearby alternate hosts where appropriate."},
    "Powdery mildew": {"about": "A fungal disease that produces characteristic white powdery growth on foliage.", "treatment": "Remove severely affected material and follow local agricultural guidance for approved controls.", "prevention": "Improve airflow and avoid crowded, humid foliage."},
    "Common rust": {"about": "A fungal rust disease that produces rust-colored pustules on crop leaves.", "treatment": "Use locally approved disease-management practices and resistant varieties where available.", "prevention": "Monitor crops regularly and maintain good field hygiene."},
    "Northern Leaf Blight": {"about": "A fungal disease of maize that can produce elongated leaf lesions.", "treatment": "Follow local extension recommendations for resistant varieties and approved controls.", "prevention": "Use resistant cultivars, crop rotation, and residue management where appropriate."},
    "Cercospora leaf spot Gray leaf spot": {"about": "A fungal leaf-spot disease of maize that can produce gray-brown lesions.", "treatment": "Follow local extension guidance for disease management and approved controls.", "prevention": "Use resistant hybrids, crop rotation, and residue management where appropriate."},
    "Esca (Black Measles)": {"about": "A grapevine disease complex associated with leaf symptoms and vine decline.", "treatment": "Consult local viticulture guidance for confirmed cases and affected vines.", "prevention": "Use healthy planting material and good vineyard sanitation."},
    "Leaf blight (Isariopsis Leaf Spot)": {"about": "A fungal leaf disease affecting grape foliage.", "treatment": "Remove affected material and follow local extension guidance.", "prevention": "Improve canopy airflow and reduce prolonged leaf wetness."},
    "Haunglongbing (Citrus greening)": {"about": "A serious citrus disease associated with leaf mottling and plant decline.", "treatment": "Seek local plant-health guidance; management focuses on infected material and vector control.", "prevention": "Use certified planting material and monitor/control disease vectors according to local guidance."},
    "Bacterial spot": {"about": "A bacterial disease that can cause small dark spots on leaves and fruit.", "treatment": "Remove severely affected material and follow local extension recommendations.", "prevention": "Reduce splash, sanitize tools, and use healthy planting material."},
    "Early blight": {"about": "A disease that commonly produces dark lesions and can spread through older leaves.", "treatment": "Remove heavily affected leaves and follow locally approved disease-management guidance.", "prevention": "Improve airflow, avoid overhead watering, remove debris, and rotate crops."},
    "Late blight": {"about": "A serious disease that can produce dark, water-soaked lesions and spread rapidly under favorable conditions.", "treatment": "Isolate affected plants and follow local agricultural guidance for approved control measures.", "prevention": "Reduce leaf wetness, improve airflow, and remove infected material promptly."},
    "Leaf Mold": {"about": "A fungal disease that can appear as pale leaf spots with mold growth under humid conditions.", "treatment": "Remove affected foliage and improve ventilation; use locally approved controls when necessary.", "prevention": "Reduce humidity around foliage and avoid prolonged leaf wetness."},
    "Septoria leaf spot": {"about": "A fungal leaf-spot disease that often produces numerous small lesions.", "treatment": "Remove affected leaves and follow local disease-management guidance.", "prevention": "Use sanitation, crop rotation, airflow, and careful watering practices."},
    "Spider mites Two-spotted spider mite": {"about": "Spider mites can cause stippling, yellowing, and general leaf decline.", "treatment": "Inspect leaf undersides and use locally approved integrated pest-management guidance if infestation is confirmed.", "prevention": "Monitor regularly and avoid plant stress."},
    "Target Spot": {"about": "A fungal leaf-spot disease that can resemble other crop spot diseases.", "treatment": "Remove affected foliage and follow local extension guidance for approved controls.", "prevention": "Improve airflow, reduce leaf wetness, and remove crop debris."},
    "Tomato Yellow Leaf Curl Virus": {"about": "A viral disease commonly associated with yellowing, curling, and stunted tomato growth.", "treatment": "There is no direct cure for an infected plant; manage vectors and follow local agricultural guidance.", "prevention": "Monitor and manage whiteflies, use healthy planting material, and control volunteer plants."},
    "Tomato mosaic virus": {"about": "A viral disease that can produce mottled or mosaic-like leaf patterns.", "treatment": "Remove severely infected plants and follow local agricultural guidance.", "prevention": "Sanitize tools, use healthy planting material, and manage vectors where applicable."},
}

# -----------------------------------------------------------------------------
# PostgreSQL
# -----------------------------------------------------------------------------
def _hash_password(password: str, salt: bytes | None = None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return salt.hex() + ":" + digest.hex()


def _verify_password(password: str, stored: str):
    try:
        salt_hex, digest_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
        return secrets.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


def _make_token(user):
    payload = {"sub": user["id"], "email": user["email"], "name": user["name"], "exp": int(__import__("time").time()) + JWT_HOURS * 3600}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_db():
    last_error = None
    for attempt in range(3):
        try:
            return psycopg2.connect(DATABASE_URL, connect_timeout=10)
        except psycopg2.Error as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2)
    raise last_error


def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    image_path TEXT,
                    image_data BYTEA,
                    image_mime TEXT,
                    plant TEXT NOT NULL,
                    disease TEXT NOT NULL,
                    raw_prediction TEXT NOT NULL DEFAULT '',
                    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'Prediction',
                    validation TEXT NOT NULL DEFAULT 'legacy',
                    created_at TEXT NOT NULL,
                    user_id TEXT
                )
            """)
            # Safe schema upgrades for databases created by an earlier version.
            cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS raw_prediction TEXT NOT NULL DEFAULT ''")
            cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS validation TEXT NOT NULL DEFAULT 'legacy'")
            cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS user_id TEXT")
            cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS image_data BYTEA")
            cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS image_mime TEXT")

            if APP_ENV != "production":
                cur.execute("SELECT id FROM users WHERE email=%s", ("demo@leafguard.ai",))
                demo = cur.fetchone()
                if not demo:
                    cur.execute(
                        """
                        INSERT INTO users (id, email, name, password_hash, created_at)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            "demo-user",
                            "demo@leafguard.ai",
                            "Demo User",
                            _hash_password("LeafGuard123!"),
                            datetime.now().isoformat(timespec="seconds"),
                        ),
                    )
                cur.execute("UPDATE predictions SET user_id='demo-user' WHERE user_id IS NULL")
        conn.commit()


def get_current_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise ValueError
    except Exception:
        raise HTTPException(status_code=401, detail="Your session has expired. Please sign in again.")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, email, name FROM users WHERE id=%s", (user_id,))
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="User account not found.")
    return {"id": row[0], "email": row[1], "name": row[2]}


init_db()

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
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
    return plant, disease


def guidance_for(disease: str):
    return GUIDANCE.get(disease, {
        "about": "The model detected a supported plant-health class.",
        "treatment": "Use local agricultural guidance before applying any treatment.",
        "prevention": "Inspect plants regularly and maintain good crop hygiene.",
    })


def image_quality(image: Image.Image):
    if image.width < 96 or image.height < 96:
        return False, "Image resolution is too small. Please upload a clearer leaf image."
    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    variance = float(gray.var())
    brightness = float(gray.mean())
    if variance < 90:
        return False, "Image appears too blurry or low-detail. Please upload a clearer leaf image."
    if brightness < 25 or brightness > 245:
        return False, "Image lighting is too dark or too bright. Please upload a well-lit leaf image."
    return True, "OK"


# -----------------------------------------------------------------------------
# Semantic leaf/non-leaf gate.
# Uses CLIP zero-shot image-text matching to reject obvious non-leaf uploads
# BEFORE the 38-class disease classifier is called. The ImageNet gate remains
# as a lightweight secondary signal.
# -----------------------------------------------------------------------------
_GATE_MODEL = None
_GATE_WEIGHTS = None
_GATE_TRANSFORM = None

SEMANTIC_MODEL_ID = os.getenv("LEAFGUARD_SEMANTIC_MODEL", "openai/clip-vit-base-patch32")
ENABLE_SEMANTIC_GATE = os.getenv("LEAFGUARD_SEMANTIC_GATE", "true").lower() == "true"
_SEMANTIC_MODEL = None
_SEMANTIC_PROCESSOR = None

LEAF_PROMPTS = [
    "a close-up photo of a plant leaf",
    "a photograph of a crop leaf",
    "a clear image of a healthy or diseased plant leaf",
    "a close-up agricultural leaf image",
]
NON_LEAF_PROMPTS = [
    "a photo of a person",
    "a portrait photo of a person",
    "a photo of an animal",
    "a photo of a vehicle",
    "a photo of a computer or electronic device",
    "a photo of an object that is not a plant",
    "a landscape or outdoor scene without a leaf as the main subject",
]

PLANT_HINTS = {
    "plant", "leaf", "flower", "fruit", "vegetable", "tree", "vine", "pot", "daisy",
    "sunflower", "rapeseed", "corn", "ear", "artichoke", "broccoli", "cabbage",
    "cauliflower", "cucumber", "bell pepper", "mushroom", "strawberry", "orange",
    "lemon", "fig", "pineapple", "banana", "pomegranate", "coffee", "jackfruit",
}
NON_PLANT_HINTS = {
    "person", "man", "woman", "boy", "girl", "face", "car", "vehicle", "laptop",
    "computer", "keyboard", "mouse", "phone", "cellular", "television", "screen",
    "chair", "dog", "cat", "horse", "bird", "bicycle", "motorcycle", "bus", "truck",
}


def _load_gate():
    global _GATE_MODEL, _GATE_WEIGHTS, _GATE_TRANSFORM
    if _GATE_MODEL is not None:
        return
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    _GATE_WEIGHTS = MobileNet_V2_Weights.DEFAULT
    _GATE_MODEL = mobilenet_v2(weights=_GATE_WEIGHTS).to(DEVICE)
    _GATE_MODEL.eval()
    _GATE_TRANSFORM = _GATE_WEIGHTS.transforms()


def _load_semantic_gate():
    global _SEMANTIC_MODEL, _SEMANTIC_PROCESSOR
    if _SEMANTIC_MODEL is not None:
        return
    _SEMANTIC_PROCESSOR = CLIPProcessor.from_pretrained(SEMANTIC_MODEL_ID)
    _SEMANTIC_MODEL = CLIPModel.from_pretrained(SEMANTIC_MODEL_ID).to(DEVICE)
    _SEMANTIC_MODEL.eval()


def _green_fraction(image: Image.Image):
    arr = np.asarray(image.resize((224, 224)).convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (g > 55) & (g > r * 1.05) & (g > b * 1.02)
    return float(mask.mean())


def _semantic_leaf_score(image: Image.Image):
    """Return leaf/non-leaf probabilities from CLIP zero-shot matching."""
    _load_semantic_gate()
    prompts = LEAF_PROMPTS + NON_LEAF_PROMPTS
    inputs = _SEMANTIC_PROCESSOR(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _SEMANTIC_MODEL(**inputs).logits_per_image[0]
        leaf_logit = logits[:len(LEAF_PROMPTS)].mean()
        nonleaf_logit = logits[len(LEAF_PROMPTS):].mean()
        probs = torch.softmax(torch.stack([leaf_logit, nonleaf_logit]), dim=0)
    return float(probs[0]), float(probs[1])


def _imagenet_gate_signal(image: Image.Image):
    """Secondary lightweight signal used when semantic scores are borderline."""
    _load_gate()
    x = _GATE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(_GATE_MODEL(x), dim=1)[0]
    values, indices = torch.topk(probs, k=5)
    labels = _GATE_WEIGHTS.meta["categories"]
    top = [(labels[int(i)], float(v)) for i, v in zip(indices, values)]
    plant_score = max((score for label, score in top if any(h in label.lower() for h in PLANT_HINTS)), default=0.0)
    nonplant_score = max((score for label, score in top if any(h in label.lower() for h in NON_PLANT_HINTS)), default=0.0)
    return top, plant_score, nonplant_score


def leaf_gate(image: Image.Image):
    """Reject obvious non-leaf uploads without loading extra validation models when disabled."""
    green = _green_fraction(image)

    # On Render Free, keep semantic validation disabled to avoid
    # downloading/loading an additional MobileNetV2 model.
    if not ENABLE_SEMANTIC_GATE:
        return True, {
            "reason": "leaf_candidate",
            "message": "Semantic validation disabled for constrained deployment.",
            "green_fraction": round(green, 3),
            "semantic_gate": "disabled",
        }

    # CLIP is used when semantic validation is enabled.
    try:
        leaf_prob, nonleaf_prob = _semantic_leaf_score(image)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The LeafGuard semantic image validator is not ready. "
                "The first run needs internet access to download its validation model. "
                f"Details: {type(exc).__name__}"
            ),
        )

    # Strong evidence that the uploaded image is not a leaf.
    if nonleaf_prob >= 0.62 and nonleaf_prob > leaf_prob:
        return False, {
            "reason": "non_leaf",
            "message": "No plant leaf was detected. Please upload a clear image of a plant leaf.",
            "leaf_probability": round(leaf_prob * 100, 2),
            "non_leaf_probability": round(nonleaf_prob * 100, 2),
            "green_fraction": round(green, 3),
        }

    # If CLIP is uncertain, use the ImageNet model as a secondary signal.
    if abs(leaf_prob - nonleaf_prob) < 0.14:
        top, plant_score, nonplant_score = _imagenet_gate_signal(image)
        if nonplant_score >= 0.35 and plant_score < 0.20 and green < 0.15:
            return False, {
                "reason": "non_leaf",
                "message": "This image does not appear to contain a plant leaf. Please upload a leaf image.",
                "leaf_probability": round(leaf_prob * 100, 2),
                "non_leaf_probability": round(nonleaf_prob * 100, 2),
                "gate_top": top,
                "green_fraction": round(green, 3),
            }

    return True, {
        "reason": "leaf_candidate",
        "message": "Semantic leaf check passed.",
        "leaf_probability": round(leaf_prob * 100, 2),
        "non_leaf_probability": round(nonleaf_prob * 100, 2),
        "green_fraction": round(green, 3),
    }


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


def gradcam(image: Image.Image, class_index: int):
    """Generate a lightweight Grad-CAM overlay for the selected class."""
    if not ENABLE_GRADCAM:
        return None
    target_layer = None
    for module in reversed(list(MODEL.modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    if target_layer is None:
        return None

    activations, gradients = [], []
    def forward_hook(_, __, output):
        activations.append(output)
    def backward_hook(_, grad_input, grad_output):
        if grad_output and grad_output[0] is not None:
            gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    try:
        inputs = PROCESSOR(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        MODEL.zero_grad(set_to_none=True)
        outputs = MODEL(**inputs).logits
        outputs[0, class_index].backward()
        if not activations or not gradients:
            return None
        acts = activations[0][0]
        grads = gradients[0][0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0).clamp(min=0)
        cam = cam / (cam.max() + 1e-8)
        cam_img = Image.fromarray(np.uint8(cam.detach().cpu().numpy() * 255)).resize(image.size, Image.Resampling.BILINEAR)
        heat = Image.new("RGB", image.size, (255, 40, 30))
        alpha = cam_img.point(lambda p: int(p * 0.55))
        overlay = Image.composite(heat, image.convert("RGB"), alpha)
        out = io.BytesIO()
        overlay.save(out, format="JPEG", quality=85)
        return base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:
        return None
    finally:
        h1.remove(); h2.remove(); MODEL.zero_grad(set_to_none=True)


@app.post("/api/auth/register")
async def register(payload: dict):
    email = str(payload.get("email", "")).strip().lower()
    name = str(payload.get("name", "")).strip()
    password = str(payload.get("password", ""))
    if not email or "@" not in email or not name or len(password) < 8:
        raise HTTPException(status_code=400, detail="Enter a valid name, email and password of at least 8 characters.")
    user_id = uuid.uuid4().hex[:12]
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO users (id, email, name, password_hash, created_at)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (user_id, email, name, _hash_password(password), datetime.now().isoformat(timespec="seconds")),
                )
            conn.commit()
    except IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    user = {"id": user_id, "email": email, "name": name}
    return {"access_token": _make_token(user), "token_type": "bearer", "user": user}


@app.post("/api/auth/login")
async def login(payload: dict):
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, email, name, password_hash FROM users WHERE email=%s", (email,))
            row = cur.fetchone()
    if not row or not _verify_password(password, row[3]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    user = {"id": row[0], "email": row[1], "name": row[2]}
    return {"access_token": _make_token(user), "token_type": "bearer", "user": user}


@app.get("/api/auth/me")
def me(user=Depends(get_current_user)):
    return user


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok", "environment": APP_ENV, "model": MODEL_ID, "architecture": MODEL.__class__.__name__,
        "device": str(DEVICE), "classes": len(CLASS_NAMES), "plants": len(SUPPORTED_PLANTS),
        "supported_plants": [DISPLAY_PLANTS.get(p, normalize_text(p)) for p in SUPPORTED_PLANTS],
        "leaf_validation": True, "gradcam": ENABLE_GRADCAM,
        "database": "postgresql",
        "rate_limiting": True,
    }


@app.get("/api/supported")
def supported(user=Depends(get_current_user)):
    grouped = {}
    for raw in CLASS_NAMES:
        plant, disease = parse_label(raw)
        grouped.setdefault(plant, []).append(disease)
    return {"classes": len(CLASS_NAMES), "plants": grouped}


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...), user=Depends(get_current_user)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a JPG, JPEG, PNG, or WEBP image.")
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Maximum image size is 5 MB.")
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded image could not be read.")

    ok, message = image_quality(image)
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    accepted, gate = leaf_gate(image)
    if not accepted:
        raise HTTPException(status_code=422, detail=gate["message"])

    top3 = predict(image)
    best = top3[0]
    confidence = best["confidence"]
    margin = confidence - top3[1]["confidence"] if len(top3) > 1 else confidence
    needs_review = confidence < REVIEW_THRESHOLD or margin < MARGIN_THRESHOLD
    status = "Needs review" if needs_review else ("Healthy" if best["disease"].lower() == "healthy" else "Disease detected")
    display_disease = best["disease"] if not needs_review else "Uncertain result"
    g = guidance_for(best["disease"])

    # Persist the accepted image inside Postgres instead of the Render filesystem.
    # Render web-service filesystems are ephemeral on the free tier, so local uploads/
    # files would disappear after restarts, redeploys, or idle spin-downs.
    record_id = uuid.uuid4().hex[:12]
    original_name = Path(file.filename or "leaf.jpg").name.replace(" ", "_")
    safe_name = f"{record_id}_{original_name}"
    stored_buffer = io.BytesIO()
    image.save(stored_buffer, format="JPEG", quality=88, optimize=True)
    stored_image = stored_buffer.getvalue()
    image_mime = "image/jpeg"
    created = datetime.now().isoformat(timespec="seconds")
    # Store every prediction under the authenticated user so records stay isolated per account.
    record = {
        "id": record_id, "filename": file.filename or safe_name, "image_path": None,
        "image_data": stored_image, "image_mime": image_mime,
        "plant": best["plant"], "disease": display_disease, "raw_prediction": best["disease"],
        "confidence": confidence, "status": status, "validation": "leaf_candidate", "created_at": created,
        "user_id": user["id"],
    }
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO predictions
                (id, filename, image_path, image_data, image_mime, plant, disease, raw_prediction, confidence, status, validation, created_at, user_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                tuple(record.values()),
            )
        conn.commit()

    image_b64 = base64.b64encode(raw_bytes).decode("ascii")
    image_mime = file.content_type or "image/jpeg"
    cam = None
    if not needs_review:
        best_idx = next((i for i, label in ID2LABEL.items() if label == top3[0]["raw"]), None)
        if best_idx is not None:
            cam = gradcam(image, best_idx)

    return {
        "id": record_id, "plant": best["plant"], "disease": display_disease,
        "raw_prediction": best["disease"], "confidence": confidence, "margin": round(margin, 2),
        "status": status, "needs_review": needs_review,
        "about": g["about"], "treatment": g["treatment"], "prevention": g["prevention"],
        "top3": [{k: x[k] for k in ("plant", "disease", "confidence")} for x in top3],
        "model_classes": len(CLASS_NAMES), "image_url": f"/api/image/{record_id}",
        "image_data": f"data:{image_mime};base64,{image_b64}", "gradcam": cam,
        "validation": gate,
    }


@app.get("/api/history")
def history(user=Depends(get_current_user)):
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT id, filename, image_path, plant, disease, raw_prediction, confidence,
                       status, validation, created_at, user_id
                       FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 50""",
                (user["id"],),
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


@app.delete("/api/history/{prediction_id}")
def delete_prediction(prediction_id: str, user=Depends(get_current_user)):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM predictions WHERE id=%s AND user_id=%s",
                (prediction_id, user["id"]),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Prediction not found")
            cur.execute(
                "DELETE FROM predictions WHERE id=%s AND user_id=%s",
                (prediction_id, user["id"]),
            )
        conn.commit()
    return {"status": "deleted"}


@app.get("/api/stats")
def stats(user=Depends(get_current_user)):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM predictions WHERE user_id=%s", (user["id"],))
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM predictions WHERE user_id=%s AND status='Healthy'", (user["id"],))
            healthy = cur.fetchone()[0]
            cur.execute("SELECT AVG(confidence) FROM predictions WHERE user_id=%s", (user["id"],))
            avg = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM predictions WHERE user_id=%s AND status='Needs review'", (user["id"],))
            reviewed = cur.fetchone()[0]
    return {
        "total": total,
        "healthy": healthy,
        "diseased": total - healthy,
        "needs_review": reviewed,
        "average_confidence": round(float(avg or 0), 2),
    }


@app.get("/api/image/{prediction_id}")
def get_image(prediction_id: str, user=Depends(get_current_user)):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT image_data, image_mime FROM predictions WHERE id=%s AND user_id=%s",
                (prediction_id, user["id"]),
            )
            row = cur.fetchone()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=bytes(row[0]), media_type=row[1] or "image/jpeg", headers={"Cache-Control": "private, max-age=300"})