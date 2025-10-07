import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--report_out', required=True)
    args = parser.parse_args()


    model = joblib.load(args.model)
    df = pd.read_csv(args.test_data)
    X = df.drop(columns=['Diabetic'])
    y = df['Diabetic']


    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else None


    report = classification_report(y, preds, output_dict=True)
    auc = roc_auc_score(y, probs) if probs is not None else None


    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    pd.Series({'report': report, 'roc_auc': auc}).to_json(args.report_out)
    print(f"Validation report saved to {args.report_out}")
