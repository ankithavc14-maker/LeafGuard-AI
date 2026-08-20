# 🌿 LeafGuard AI

**AI-powered plant disease detection with FastAPI, deep learning, JWT authentication, PostgreSQL, and an interactive web interface.**

**Live Demo:** https://leafguard77.onrender.com/

LeafGuard AI is a production-oriented computer-vision application that accepts plant-leaf images, validates the input, runs an image-classification model, returns top predictions with confidence scores, and stores authenticated prediction history and analytics.

## ✨ Features

- **Plant disease classification** using a Hugging Face image-classification model.
- **FastAPI inference API** with image upload and prediction endpoints.
- **Top-3 predictions** with confidence scores.
- **Input validation** for file type, file size, image dimensions, blur/quality, and plant-leaf suitability.
- **JWT authentication** for protected prediction/history endpoints.
- **Prediction history and statistics** backed by PostgreSQL.
- **Grad-CAM visualization** for model explainability when enabled.
- **Rate limiting** on authentication and prediction routes.
- **Security headers** and production HTTPS hardening.
- **Responsive frontend** served directly by FastAPI.
- **Multilingual interface** support.

## 🧠 Model & Evaluation

The application is configured through `LEAFGUARD_MODEL` and currently defaults to:

```text
Kathir56/plant-disease-tamilnadu
```

The model and supported labels are loaded dynamically at application startup.

The project documentation includes evaluation material, including a confusion matrix and model-evaluation script under `docs/` and `scripts/`.

## 🏗️ Architecture

```text
User
  │
  ▼
LeafGuard Web UI
  │
  ▼
FastAPI Application
  ├── JWT Authentication
  ├── Image Validation
  ├── ML Inference
  ├── Grad-CAM
  ├── Prediction History
  └── Statistics
          │
          ▼
      PostgreSQL
```

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| Language | Python |
| API | FastAPI, Uvicorn |
| Deep Learning | PyTorch, Transformers |
| Image Processing | Pillow, NumPy |
| ML Evaluation | scikit-learn, Matplotlib |
| Authentication | PyJWT |
| Database | PostgreSQL |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render |
| Configuration | python-dotenv |

## 📁 Project Structure

```text
LeafGuard-AI/
├── app.py
├── requirements.txt
├── render.yaml
├── .env.example
├── .python-version
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── scripts/
│   └── evaluate_model.py
├── docs/
│   ├── assets/
│   │   └── confusion_matrix.png
│   └── project/deployment notes
└── notebooks/
    └── leafguard_experiments.ipynb
```

## 🚀 Run Locally

### 1. Clone

```bash
git clone https://github.com/ankithavc14-maker/LeafGuard-AI.git
cd LeafGuard-AI
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and set:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/leafguard
SECRET_KEY=replace-with-a-random-secret-of-at-least-32-characters
APP_ENV=development
LEAFGUARD_GRADCAM=true
```

Optional model overrides:

```env
LEAFGUARD_MODEL=Kathir56/plant-disease-tamilnadu
LEAFGUARD_SEMANTIC_MODEL=openai/clip-vit-base-patch32
```

### 5. Start the API

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## ☁️ Deployment

The repository includes `render.yaml` for the Render deployment configuration. The service is configured to run:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Production configuration requires a strong `SECRET_KEY` and a PostgreSQL `DATABASE_URL`.

## 🔐 Security Notes

- Never commit `.env` or production secrets.
- Use a strong random `SECRET_KEY` in production.
- Prediction and authentication routes have lightweight per-process rate limiting.
- Protected API routes use JWT bearer authentication.
- Uploaded image storage and database configuration should be treated as production infrastructure concerns.

## ⚠️ Medical / Agricultural Disclaimer

LeafGuard AI is a **decision-support tool**, not a substitute for professional agricultural diagnosis. Uncertain predictions and treatment options should be verified with qualified local agricultural guidance.

## 👩‍💻 Author

**Ankitha V Chandan**

GitHub: https://github.com/ankithavc14-maker
