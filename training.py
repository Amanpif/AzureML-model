import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score




def train(input_path: str, model_out: str, metrics_out: str, n_estimators: int = 100, random_state: int = 42):
    df = pd.read_csv(input_path)
    X = df.drop(columns=['Diabetic'])
    y = df['Diabetic']


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)


    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)


    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:,1] if hasattr(model, 'predict_proba') else None


    report = classification_report(y_val, preds, output_dict=True)
    auc = roc_auc_score(y_val, probs) if probs is not None else None


    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)


    metrics = { 'classification_report': report }
    if auc is not None:
        metrics['roc_auc'] = float(auc)


    pd.Series(metrics).to_json(metrics_out)
    print(f"Model saved to {model_out}")
    print(f"Metrics saved to {metrics_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model_out', required=True)
    parser.add_argument('--metrics_out', required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    args = parser.parse_args()
    train(args.input, args.model_out, args.metrics_out, args.n_estimators)