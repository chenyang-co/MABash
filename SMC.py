import os
import re
import pandas as pd

# =======================
# Path configuration
# =======================
DATASET_DIR = "/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/dataset"
INPUT_FILE = os.path.join(DATASET_DIR, "test.csv")

OUTPUT_SIMPLE = os.path.join(DATASET_DIR, "test_simple.csv")
OUTPUT_MEDIUM = os.path.join(DATASET_DIR, "test_medium.csv")
OUTPUT_COMPLEX = os.path.join(DATASET_DIR, "test_complex.csv")


# =======================
# Structural feature extraction
# =======================
def extract_structure_features(cmd: str):
    return {
        "num_pipes": cmd.count("|"),
        "num_redirections": len(re.findall(r"(>>|<<|>|<)", cmd)),
        "num_exec": cmd.count("-exec"),
        "num_subshell": len(re.findall(r"\$\(|`", cmd)),
        "num_logical": len(re.findall(r"&&|\|\|", cmd)),
    }


# =======================
# Complexity classification
# =======================
def classify_complexity(cmd: str) -> str:
    feat = extract_structure_features(cmd)

    # Complex commands
    if (
        feat["num_pipes"] >= 2
        or (feat["num_pipes"] >= 1 and feat["num_redirections"] >= 1)
        or feat["num_exec"] >= 2
        or feat["num_subshell"] > 0
        or feat["num_logical"] > 0
    ):
        return "Complex"

    # Medium commands
    if feat["num_pipes"] == 1 or feat["num_redirections"] == 1:
        return "Medium"

    # Simple commands
    return "Simple"


# =======================
# Dataset splitting
# =======================
def split_dataset():
    # Load dataset
    df = pd.read_csv(INPUT_FILE)

    assert "code" in df.columns and "nl" in df.columns, \
        "CSV must contain 'code' and 'nl' columns"

    simple_rows = []
    medium_rows = []
    complex_rows = []

    for _, row in df.iterrows():
        cmd = str(row["code"]).strip()
        level = classify_complexity(cmd)

        if level == "Simple":
            simple_rows.append(row)
        elif level == "Medium":
            medium_rows.append(row)
        else:
            complex_rows.append(row)

    # Convert to DataFrame
    df_simple = pd.DataFrame(simple_rows)
    df_medium = pd.DataFrame(medium_rows)
    df_complex = pd.DataFrame(complex_rows)

    # Save datasets
    df_simple.to_csv(OUTPUT_SIMPLE, index=False)
    df_medium.to_csv(OUTPUT_MEDIUM, index=False)
    df_complex.to_csv(OUTPUT_COMPLEX, index=False)

    # Print statistics (for experiment logging)
    print("Dataset split completed:")
    print(f"  Simple : {len(df_simple)} samples")
    print(f"  Medium : {len(df_medium)} samples")
    print(f"  Complex: {len(df_complex)} samples")


# =======================
# Main
# =======================
if __name__ == "__main__":
    split_dataset()
