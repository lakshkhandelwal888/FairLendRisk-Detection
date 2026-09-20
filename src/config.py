"""Project settings for FairLend. Change the policy here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "processed" / "clean.csv"
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"

TARGET = "Status"          # 1 = defaulted, 0 = did not default
TEST_SIZE = 0.2
RANDOM_STATE = 42          # fixed seed so the split is identical on every machine

# Columns removed before modelling, with the reason for each.
DROP_COLUMNS = {
    "ID": "row identifier, no predictive meaning",
    "year": "constant value (2019)",
    "rate_of_interest": "leakage: filled in only for loans that did not default",
    "Interest_rate_spread": "leakage: filled in only for loans that did not default",
    "Upfront_charges": "leakage: filled in only for loans that did not default",
    "credit_type": "leakage: category EQUI is 100% default; other categories carry no signal",
    "property_value": "leakage: missing in 100% of rows that defaulted",
    "LTV": "leakage: missing in 100% of rows that defaulted",
    "dtir1": "leakage: missing far more often for defaulted loans (68% vs 16%)",
}

# Rows removed because they are data-recording artifacts.
#   age missing -> 200 rows, all defaulted
#   income == 0 -> 1,260 rows, 99.4% defaulted

# Sensitive attributes: kept in the data for the fairness audit later.
SENSITIVE_COLS = ["Gender", "age"]


# Columns kept in train/test files but NOT given to the model as inputs.
# Gender stays available for the fairness audit later.
MODEL_EXCLUDE = ["Gender"]

# Experiment tracking
MLFLOW_DB = ROOT / "mlflow.db"          # MLflow stores run info in this small database file
EXPERIMENT_NAME = "fairlend-baseline"
METRICS_JSON = ROOT / "reports" / "metrics.json"