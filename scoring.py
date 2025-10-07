import json
import joblib
import pandas as pd
import os


class ScoringService:
    def __init__(self):
        self.model = None


    def init(self, model_path):
        self.model = joblib.load(model_path)


    def run(self, request_json):
# request_json: list of records or dict
        df = pd.DataFrame(request_json if isinstance(request_json, list) else [request_json])
        preds = self.model.predict(df)
        return json.dumps({'predictions': preds.tolist()})


# Local test
if __name__ == '__main__':
    import sys
    mpath = sys.argv[1]
    scoring = ScoringService()
    scoring.init(mpath)
    sample = {'age': 50}
    print(scoring.run(sample))