# LeafGuard AI — Phases 1–3 completed

## Phase 1: Prediction reliability
- 38-class plant-health model retained.
- File type and 5 MB size validation.
- Minimum image resolution validation.
- Basic blur/detail and brightness checks.
- Lazy ImageNet MobileNetV2 plant/non-plant gate for obvious non-leaf inputs.
- Low-confidence and low-margin predictions are marked **Needs review**.
- Invalid/non-leaf uploads are rejected before prediction history is saved.

> The leaf gate is a practical validation safeguard, not a certified leaf detector or true OOD model.

## Phase 2: ML evaluation and explainability
- Added `evaluate_model.py` for independent test-set evaluation.
- Reports accuracy, precision, recall and F1.
- Saves a confusion matrix.
- Added Grad-CAM for confident predictions.
- Top-3 alternatives are displayed.

Run evaluation only when you have a properly labeled test set:

```bash
python evaluate_model.py --data ./test_dataset
```

Do not claim a new accuracy number without running this evaluation.

## Phase 3: Product features
- Professional responsive frontend.
- Dashboard and statistics.
- Prediction history.
- Delete prediction.
- Plant Guide generated from the model's actual class list.
- Symptoms/treatment/prevention guidance.
- Downloadable text prediction report.
- User registration and JWT login.
- Per-user prediction history.
- Demo account included for presentations.

### Demo account
- Email: `demo@leafguard.ai`
- Password: `LeafGuard123!`

Change the JWT secret before production deployment by setting:

```text
LEAFGUARD_JWT_SECRET=<strong-random-secret>
```

## Current scope
This version supports the 38 classes provided by the selected model. It does **not** recognize every plant species in existence. Unsupported/uncertain inputs should be treated as review cases.
