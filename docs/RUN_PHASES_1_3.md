# LeafGuard AI — Run Guide (Phases 1–3)

## 1. Create/activate environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

If your existing `.venv` already works, reuse it.

## 2. Install packages

```powershell
python -m pip install -r requirements.txt
```

## 3. Run

```powershell
python -m uvicorn app:app
```

Open:

`http://127.0.0.1:8000`

## Demo account

- Email: `demo@leafguard.ai`
- Password: `LeafGuard123!`

The first prediction may take longer because the 38-class Hugging Face model and the lazy ImageNet validation model may need to download/cache.

## Test cases

1. Supported healthy/diseased leaf → prediction + confidence + Grad-CAM.
2. Person/car/object photo → should be rejected by the leaf validation gate.
3. Blurry/dark/bright image → should be rejected by image-quality validation.
4. Low-confidence/low-margin leaf → marked `Needs review`.
5. Sign out → protected pages require login again.

## Evaluation

When you have a properly labeled test set with the same 38 class folder names:

```powershell
python evaluate_model.py --data .\test_dataset
```

This produces accuracy, precision, recall, F1 and a confusion matrix. Do not report an accuracy number until this independent evaluation has been run.
