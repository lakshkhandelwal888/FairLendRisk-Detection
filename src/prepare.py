"""Apply the leakage policy to clean.csv and split it into train/test sets."""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (CLEAN_CSV, TRAIN_CSV, TEST_CSV, TARGET, TEST_SIZE,
                    RANDOM_STATE, DROP_COLUMNS)


def main():
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded clean data: {len(df):,} rows x {df.shape[1]} columns")

    # 1. Remove artifact rows
    age_missing = df["age"].isna()
    zero_income = df["income"] == 0
    print(f"Rows with missing age:   {age_missing.sum():,}")
    print(f"Rows with income == 0:   {zero_income.sum():,}")
    df = df[~(age_missing | zero_income)]

    # 2. Remove leaky / useless columns
    df = df.drop(columns=list(DROP_COLUMNS))
    print(f"After policy: {len(df):,} rows x {df.shape[1]} columns "
          f"(default rate {df[TARGET].mean():.3f})")

    # 3. Stratified split: both sets keep the same default rate
    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"Train: {len(train):,} rows (default rate {train[TARGET].mean():.3f})")
    print(f"Test:  {len(test):,} rows (default rate {test[TARGET].mean():.3f})")


if __name__ == "__main__":
    main()