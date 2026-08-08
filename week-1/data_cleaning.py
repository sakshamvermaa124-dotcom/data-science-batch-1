"""
Week 1 Task: Data Cleaning with Pandas
"""
import pandas as pd
import numpy as np

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw dataset: handle nulls, fix dtypes, remove duplicates."""
    df = df.copy()

    # 1. Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {before - len(df)} duplicate rows")

    # 2. Handle missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)

    # 3. Remove outliers (IQR method)
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR     = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

    print(f"Clean dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(0)
    raw = pd.DataFrame({
        "x": np.append(np.random.normal(0, 1, 100), [999, -999]),  # outliers
        "y": np.random.choice(["A","B","C", None], 102),
    })
    cleaned = clean_dataset(raw)
    print(cleaned.describe())
