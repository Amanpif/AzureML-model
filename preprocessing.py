import argparse
import pandas as pd
import os


def preprocess(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    # Basic cleaning: drop rows with null target, drop duplicated rows
    df = df.dropna(subset=["Diabetic"])
    df = df.drop_duplicates()


    # Example: simple imputation for numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Ensure target numeric
    if df['Diabetic'].dtype == 'object':
        df['Diabetic'] = pd.factorize(df['Diabetic'])[0]


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    preprocess(args.input, args.output)