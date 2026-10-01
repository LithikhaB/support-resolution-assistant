from pathlib import Path

import pandas as pd
from datasets import load_dataset


RAW_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DIR / "customer_support_tickets.csv"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Hugging Face dataset...")
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets")

    train = dataset["train"]

    print(f"Total rows: {len(train)}")

    df = train.to_pandas()

    # Keep English tickets for the initial system.
    df = df[df["language"].str.lower() == "en"].copy()

    print(f"English rows: {len(df)}")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved dataset to: {OUTPUT_FILE}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()