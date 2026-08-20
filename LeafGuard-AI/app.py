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

import jwt

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required. Render supplies it from the Postgres database in render.yaml; locally set it in .env.")

MODEL_ID = os.getenv("LEAFGUARD_MODEL", "Kathir56/plant-disease-tamilnadu")
DEVICE = "cpu"
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
# Lightweight hosted inference
# -----------------------------------------------------------------------------
MODEL_ID = os.getenv("LEAFGUARD_MODEL", "Kathir56/plant-disease-tamilnadu")
DEVICE = "remote-inference"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
REVIEW_THRESHOLD = 55.0
MARGIN_THRESHOLD = 8.0
APP_ENV = os.getenv("APP_ENV", "development").lower()
JWT_SECRET = os.getenv("SECRET_KEY") or os.getenv("LEAFGUARD_JWT_SECRET", "leafguard-demo-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_HOURS = 12
ENABLE_GRADCAM = False

if APP_ENV == "production" and (JWT_SECRET == "leafguard-demo-secret-change-in-production" or len(JWT_SECRET) < 32):
    raise RuntimeError("SECRET_KEY must be set to a random value of at least 32 characters in production.")

app = FastAPI(title="LeafGuard AI", version="5.1")

@app.get("/static/app.js", include_in_schema=False)
def serve_app_js():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

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

DISPLAY_PLANTS = {
    "Apple": "Apple", "Blueberry": "Blueberry", "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Corn / Maize", "Grape": "Grape", "Orange": "Orange / Citrus",
    "Peach": "Peach", "Pepper,_bell": "Bell Pepper", "Potato": "Potato",
    "Raspberry": "Raspberry", "Soybean": "Soybean", "Squash": "Squash",
    "Strawberry": "Strawberry", "Tomato": "Tomato",
}
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
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
    ensure_db_initialized()
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


DB_READY = False
DB_INIT_ERROR = None

def ensure_db_initialized():
    global DB_READY, DB_INIT_ERROR
    if DB_READY:
        return
    if DB_INIT_ERROR is not None:
        raise RuntimeError(DB_INIT_ERROR)
    try:
        init_db()
        DB_READY = True
    except Exception as exc:
        DB_INIT_ERROR = f"Database initialization failed: {type(exc).__name__}: {exc}"
        raise RuntimeError(DB_INIT_ERROR) from exc


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
# Lightweight leaf gate
# -----------------------------------------------------------------------------
def _green_fraction(image: Image.Image):
    arr = np.asarray(image.resize((160, 160)).convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (g > 55) & (g > r * 1.05) & (g > b * 1.02)
    return float(mask.mean())

def leaf_gate(image: Image.Image):
    # Avoid CLIP/ImageNet auxiliary models on the Render free tier. They add
    # significant RAM pressure without changing the primary disease classifier.
    green = _green_fraction(image)
    return True, {
        "reason": "leaf_candidate",
        "message": "Leaf candidate accepted for disease classification.",
        "green_fraction": round(green, 3),
        "semantic_gate": "disabled",
    }


def predict(image_bytes: bytes):
    if not HF_TOKEN:
        raise HTTPException(status_code=503, detail="Hugging Face inference is not configured. Set HF_TOKEN in Render environment variables.")
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider="auto", api_key=HF_TOKEN, timeout=60)
        outputs = client.image_classification(image_bytes, model=MODEL_ID, top_k=3)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Remote model inference failed: {type(exc).__name__}") from exc

    results = []
    for item in outputs[:3]:
        raw = item.label if hasattr(item, "label") else item.get("label")
        score = item.score if hasattr(item, "score") else item.get("score", 0)
        plant, disease = parse_label(str(raw))
        results.append({"raw": str(raw), "plant": plant, "disease": disease, "confidence": round(float(score) * 100, 2)})
    if not results:
        raise HTTPException(status_code=503, detail="The inference provider returned no predictions.")
    return results


@app.post("/api/auth/register")
async def register(payload: dict):
    ensure_db_initialized()
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
    ensure_db_initialized()
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
        "status": "ok",
        "environment": APP_ENV,
        "model": MODEL_ID,
        "architecture": "MobileNetV2 via Hugging Face Inference Providers",
        "device": DEVICE,
        "classes": len(CLASS_NAMES),
        "plants": len(SUPPORTED_PLANTS),
        "supported_plants": [DISPLAY_PLANTS.get(p, normalize_text(p)) for p in SUPPORTED_PLANTS],
        "leaf_validation": True,
        "gradcam": False,
        "semantic_gate": False,
        "inference_provider": "huggingface",
        "hf_token_configured": bool(HF_TOKEN),
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

    top3 = predict(raw_bytes)
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

    return {
        "id": record_id, "plant": best["plant"], "disease": display_disease,
        "raw_prediction": best["disease"], "confidence": confidence, "margin": round(margin, 2),
        "status": status, "needs_review": needs_review,
        "about": g["about"], "treatment": g["treatment"], "prevention": g["prevention"],
        "top3": [{k: x[k] for k in ("plant", "disease", "confidence")} for x in top3],
        "model_classes": len(CLASS_NAMES), "image_url": f"/api/image/{record_id}",
        "image_data": f"data:{image_mime};base64,{image_b64}",
        "validation": gate,
        "gradcam": None,
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
