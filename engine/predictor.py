import joblib
import pandas as pd

class AMRPredictor:

    def __init__(self, model, threshold, feature_columns, cat_cols):
        self.model = model
        self.threshold = threshold
        self.feature_columns = feature_columns
        self.cat_cols = cat_cols

    def preprocess(self, df):
        df = df[self.feature_columns]
        for col in self.cat_cols:
            df[col] = df[col].astype("category")
        return df

    def predict_resistance(self, df):
        df = self.preprocess(df)
        return self.model.predict_proba(df)[:, 1]

    def predict(self, df):
        probs = self.predict_resistance(df)
        return (probs >= self.threshold).astype(int)