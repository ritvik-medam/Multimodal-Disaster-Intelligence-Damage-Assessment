"""
Master Training Orchestrator for Multimodal Disaster Intelligence System
Executes training for:
1. Disaster Image Classifier (EfficientNet-B2)
2. Disaster Text Classifier (DistilBERT)
3. Satellite Damage Assessment (ResNet-50)
4. Multimodal Fusion Model (Dual-Encoder)
Generates unified evaluation benchmark summaries across all trained models.
"""

import sys
import time
from pathlib import Path

from training.train_disaster_classifier import train_disaster_classifier
from training.train_text_classifier import train_text_classifier
from training.train_damage_assessment import train_damage_assessment
from training.train_multimodal_model import train_multimodal_model

ROOT_DIR = Path(__file__).resolve().parent.parent


def run_full_training_pipeline(epochs_override: int = None):
    print("\n" + "=" * 70)
    print("🚀 LAUNCHING FULL MULTIMODAL DISASTER INTELLIGENCE TRAINING PIPELINE")
    print("=" * 70)
    start_time = time.time()

    results = {}

    # 1. Image Model
    print("\n[Step 1/4] Training Disaster Image Classifier (EfficientNet-B2)...")
    results["image_model"] = train_disaster_classifier(
        str(ROOT_DIR / "configs/image_model.yaml"),
        max_epochs_override=epochs_override or 6
    )

    # 2. Text Model
    print("\n[Step 2/4] Fine-Tuning Disaster Text Classifier (DistilBERT)...")
    results["text_model"] = train_text_classifier(
        str(ROOT_DIR / "configs/text_model.yaml"),
        max_epochs_override=epochs_override or 3
    )

    # 3. Damage Assessment
    print("\n[Step 3/4] Training Satellite Building Damage Assessment (ResNet-50)...")
    results["damage_model"] = train_damage_assessment(
        str(ROOT_DIR / "configs/damage_model.yaml"),
        max_epochs_override=epochs_override or 5
    )

    # 4. Multimodal Model
    print("\n[Step 4/4] Training Multimodal Dual-Encoder Fusion Model...")
    results["multimodal_model"] = train_multimodal_model(
        str(ROOT_DIR / "configs/multimodal_model.yaml"),
        max_epochs_override=epochs_override or 3
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 ALL 4 MODELS TRAINED SUCCESSFULLY IN {elapsed/60:.2f} MINUTES!")
    print("=" * 70)
    print("\n📊 FINAL TEST SET BENCHMARK SUMMARY:")
    print(f"{'Model':<30} | {'Accuracy':<10} | {'Macro F1':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 75)
    for model_name, m in results.items():
        print(f"{model_name:<30} | {m.get('accuracy',0):<10.4f} | {m.get('f1_macro',0):<10.4f} | {m.get('precision_macro',0):<10.4f} | {m.get('recall_macro',0):<10.4f}")
    print("-" * 75)


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_full_training_pipeline(epochs_override=epochs)
