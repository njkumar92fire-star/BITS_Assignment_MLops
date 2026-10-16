"""Central configuration: paths, column groups and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_clean.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
FIGURES_DIR = ROOT / "reports" / "figures"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# Column groups used by the preprocessing pipeline
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ENGINEERED_FEATURES = ["hr_ratio"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MLFLOW_EXPERIMENT = "heart-disease-risk"
MLFLOW_MODEL_NAME = "heart-disease-classifier"
