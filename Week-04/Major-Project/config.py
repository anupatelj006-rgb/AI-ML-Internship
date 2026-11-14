from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models_saved"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]

DATASET_NAME = "TrashNet (5 of 6 classes; 'trash' excluded)"
DATASET_SOURCE = "Yang and Thung, 2016. https://github.com/garythung/trashnet"

IMG_SIZE = 128
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

SEED = 42
SPLIT = (0.70, 0.15, 0.15)

BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 3
NUM_WORKERS = 0

MODEL_NAMES = ["baseline", "improved", "transfer"]
ENSEMBLE = "ensemble"
AUGMENT = {"baseline": False, "improved": True, "transfer": True}
MODEL_TITLES = {
    "baseline": "Baseline CNN",
    "improved": "Improved CNN (BatchNorm + Dropout + Augmentation)",
    "transfer": "Transfer Learning (ResNet18, fine-tuned)",
    "ensemble": "Ensemble (average of all models)",
}

CONFIDENCE_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.15