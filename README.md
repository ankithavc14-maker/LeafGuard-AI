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
