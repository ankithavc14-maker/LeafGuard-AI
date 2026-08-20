"""Evaluate LeafGuard's 38-class image classifier on an ImageFolder dataset.

Expected structure:
  dataset/
    Apple___Apple_scab/
      image1.jpg
    Apple___Black_rot/
      image2.jpg
    ...

Usage:
  python evaluate_model.py --data ./test_dataset

The script reports accuracy, macro precision/recall/F1 and saves a confusion matrix.
It does NOT claim that the model's public model-card accuracy is the project's own evaluation.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='ImageFolder test dataset directory')
    ap.add_argument('--model', default='Kathir56/plant-disease-tamilnadu')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--output', default='evaluation_confusion_matrix.png')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForImageClassification.from_pretrained(args.model).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model)

    dataset = ImageFolder(args.data)
    labels = dataset.classes

    def collate(batch):
        images, targets = zip(*batch)
        inputs = processor(images=list(images), return_tensors='pt')
        return inputs, torch.tensor(targets)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model_labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    n = len(model_labels)
    if len(labels) != n:
        raise SystemExit(f'Dataset has {len(labels)} folders but model has {n} classes. Use the same 38 class names as the model.')
    model_index_by_name = {name: i for i, name in enumerate(model_labels)}
    missing = [name for name in labels if name not in model_index_by_name]
    if missing:
        raise SystemExit('Dataset class folders do not match model labels. Missing from model: ' + ', '.join(missing[:8]))

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model_preds = model(**inputs).logits.argmax(dim=1).cpu().numpy()
            y_pred.extend([labels.index(model_labels[int(p)]) for p in model_preds])
            y_true.extend(targets.numpy().tolist())

    print(f'Images: {len(y_true)}')
    print(f'Accuracy: {accuracy_score(y_true, y_pred):.4f}')
    print(classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    fig, ax = plt.subplots(figsize=(15, 13))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=90, colorbar=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f'Confusion matrix saved to: {args.output}')


if __name__ == '__main__':
    main()
