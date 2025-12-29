import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../..", "output")

# --- INPUT: paths to your CSVs ---
CSV_PATHS = {
    100:  os.path.join(DATA_DIR, "workers-time-100.csv"),
    500:  os.path.join(DATA_DIR, "workers-time-500.csv"),
    1000: os.path.join(DATA_DIR, "workers-time-1000.csv"),
    2000: os.path.join(DATA_DIR, "workers-time-2000.csv"),
    3000: os.path.join(DATA_DIR, "workers-time-3000.csv"),
    5000: os.path.join(DATA_DIR, "workers-time-5000.csv"),
}