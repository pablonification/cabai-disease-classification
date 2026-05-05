from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
REPORTS_DIR = OUTPUT_DIR / "reports"

CLASS_NAMES = ["healthy", "leaf curl", "leaf spot", "whitefly", "yellowish"]
EXCLUDED_CLASSES = ["powdery mildew"]

DISPLAY_NAMES = {
    "healthy": "sehat",
    "leaf curl": "keriting daun",
    "leaf spot": "bercak daun",
    "whitefly": "whitefly / kutu kebul",
    "yellowish": "virus kuning / daun menguning",
}

LABEL_ALIASES = {
    "healthy": "healthy",
    "sehat": "healthy",
    "cabai sehat": "healthy",
    "leaf curl": "leaf curl",
    "leafcurl": "leaf curl",
    "curl": "leaf curl",
    "keriting daun": "leaf curl",
    "leaf spot": "leaf spot",
    "leafspot": "leaf spot",
    "bercak daun": "leaf spot",
    "whitefly": "whitefly",
    "white fly": "whitefly",
    "kutu kebul": "whitefly",
    "yellowish": "yellowish",
    "yellow": "yellowish",
    "virus kuning": "yellowish",
    "daun menguning": "yellowish",
    "powdery mildew": "powdery mildew",
}

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "val",
    "validation": "val",
    "val": "val",
    "test": "test",
    "testing": "test",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RANDOM_SEED = 42
