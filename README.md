# 🌿 LeafGuard AI — Plant Disease Classifier

> Deep learning model that identifies plant leaf diseases from a photo, deployed as a live API.

---

## Overview

LeafGuard AI uses transfer learning (MobileNetV2) to classify plant leaf images into 15 categories — covering diseases and healthy leaves across pepper, potato, and tomato plants. Trained on the PlantVillage dataset (~20,600 images), it achieves **94% test accuracy** and is deployed as a FastAPI inference service.

---

## Features

| Feature | Details |
|---|---|
| 🧠 **Transfer Learning** | Fine-tuned MobileNetV2 (pretrained on ImageNet) |
| 🌱 **15 Classes** | Pepper, potato, and tomato — disease + healthy categories |
| 📊 **94% Test Accuracy** | Evaluated on a held-out test set (3,109 images) |
| 🚀 **REST API** | FastAPI endpoint — upload an image, get a diagnosis |
| 📈 **Full Evaluation** | Precision/recall/F1 per class + confusion matrix |

---

## Results

**Test Accuracy: 94%** across 15 classes (3,109 test images)

Strongest performance on `Pepper_bell_healthy` and `Tomato_YellowLeaf_Curl_Virus` (99–100% F1). Most confusion occurs between visually similar tomato leaf-spot diseases (e.g. `Early_blight` vs `Target_Spot`), which is consistent with how subtle these distinctions are even to the human eye.

![Confusion Matrix](assets/confusion_matrix.png)

---

## Tech Stack

-


UI update: the application now uses a single upload area on the Dashboard. The Predict Disease navigation item focuses that same upload area instead of opening a second upload form.


## Render deployment
This project includes a Render Blueprint (`render.yaml`) that creates a Free Render Web Service and a Free Render Postgres database, and automatically wires `DATABASE_URL` to the app. Accepted prediction images are stored in Postgres so they do not depend on Render's ephemeral filesystem. Free Render Web Services sleep after inactivity, and Free Render Postgres expires after 30 days; these are platform limitations, not application errors.
