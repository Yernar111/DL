import torch
from pathlib import Path


# =========================
# Paths
# =========================

# DATA_DIR = "data"
DATA_DIR = Path("data")

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_CSV = DATA_DIR / "Training_set.csv"
TEST_CSV = DATA_DIR / "Testing_set.csv"


# MODEL_DIR = "models"
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "best_model.pth"
CLASS_MAPPING_PATH = MODEL_DIR / "class_to_idx.json"
LOSS_HISTORY_PATH = MODEL_DIR / "loss_history.json"
METRICS_HISTORY_PATH = MODEL_DIR / "metrics_history.json"


# =========================
# Dataset
# =========================

IMAGE_SIZE = 224
NUM_CLASSES = 75
BATCH_SIZE = 64
TRAIN_SIZE = 0.8

# =========================
# Training
# =========================

NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
PATIENCE = 3


# =========================
# etc
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
IMG_PATH = "data1"


