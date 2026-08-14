# Semantic Leaf Validation

LeafGuard AI now performs a semantic image check before the 38-class disease model runs.

## How it works

1. Basic file/type/size/quality validation runs first.
2. CLIP (`openai/clip-vit-base-patch32`) compares the image against leaf and non-leaf text prompts.
3. Obvious non-leaf images (people, vehicles, electronics, animals, etc.) are rejected with HTTP 422.
4. Borderline cases receive a secondary ImageNet MobileNetV2 signal.
5. Only accepted images are sent to the plant-disease classifier.

## First run

The semantic model is downloaded from Hugging Face on the first prediction. Internet access is required for that first download. Later runs use the local Hugging Face cache.

You can change the model with:

```powershell
$env:LEAFGUARD_SEMANTIC_MODEL="openai/clip-vit-base-patch32"
```

## Important limitation

This is a semantic validation gate, not a formally trained leaf-segmentation or object-detection model. It is designed to reject obvious invalid uploads and reduce forced classifications. It should still be evaluated on a dedicated leaf/non-leaf validation set before being described as production-grade.
