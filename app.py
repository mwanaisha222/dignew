from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from engine.predictor import AMRPredictor
from engine.recommender import AntibioticRecommender
from engine.rules import ClinicalRules

# ------------------------
# Initialize App
# ------------------------
app = FastAPI(title="AMR Clinical Decision Support API")

# ------------------------
# Load Production Artifacts
# ------------------------
model = joblib.load("models/tuned_lgbm.pkl")
threshold = joblib.load("models/threshold.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
antibiotics = joblib.load("models/antibiotics.pkl")
cat_cols = joblib.load("models/cat_cols.pkl")

# ------------------------
# Initialize Engine
# ------------------------
predictor = AMRPredictor(
    model=model,
    threshold=threshold,
    feature_columns=feature_columns,
    cat_cols=cat_cols
)

recommender = AntibioticRecommender(predictor, antibiotics)

rules = ClinicalRules(
    reserve_drugs=["Meropenem"]
)

# ------------------------
# Patient Input Schema
# ------------------------
class PatientInput(BaseModel):
    species: str
    family: str
    country: str
    age_group: str
    source: str
    year: int
    phenotype: str
    in_out_patient: str
    dataset: str


# ------------------------
# Recommendation Endpoint
# ------------------------
@app.post("/recommend")
def recommend_antibiotics(patient: PatientInput):

    patient_dict = patient.dict()

    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_dict])

    # Generate ranked recommendations
    df = recommender.recommend(patient_df)

    # Apply clinical rules
    df = rules.apply(df)

    return {
        "recommendations": df.to_dict(orient="records")
    }