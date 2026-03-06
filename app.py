from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional
import pandas as pd
import joblib

from engine.predictor import AMRPredictor
from engine.recommender import AntibioticRecommender
from engine.rules import ClinicalRules
from engine.valid_values import (
    VALID_FAMILY, VALID_COUNTRY, VALID_AGE_GROUP, VALID_SOURCE,
    VALID_PHENOTYPE, VALID_DATASET, MIN_YEAR, MAX_YEAR,
)

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

# Extract valid species from the trained model (362 values — too many to hardcode)
VALID_SPECIES = set(sorted(model.booster_.pandas_categorical[0]))

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
    phenotype: Optional[str] = "unknown"
    in_out_patient: str = "unknown"
    dataset: str

    @field_validator("species")
    @classmethod
    def validate_species(cls, v):
        if v not in VALID_SPECIES:
            raise ValueError(f"Unknown species '{v}'. Must be a recognised bacterial species.")
        return v

    @field_validator("family")
    @classmethod
    def validate_family(cls, v):
        if v not in VALID_FAMILY:
            raise ValueError(f"Unknown family '{v}'. Must be one of: {VALID_FAMILY}")
        return v

    @field_validator("country")
    @classmethod
    def validate_country(cls, v):
        if v not in VALID_COUNTRY:
            raise ValueError(f"Unknown country '{v}'. Must be a supported country.")
        return v

    @field_validator("age_group")
    @classmethod
    def validate_age_group(cls, v):
        if v not in VALID_AGE_GROUP:
            raise ValueError(f"Unknown age_group '{v}'. Must be one of: {VALID_AGE_GROUP}")
        return v

    @field_validator("source")
    @classmethod
    def validate_source(cls, v):
        if v not in VALID_SOURCE:
            raise ValueError(f"Unknown source '{v}'. Must be a recognised specimen source.")
        return v

    @field_validator("year")
    @classmethod
    def validate_year(cls, v):
        if not (MIN_YEAR <= v <= MAX_YEAR):
            raise ValueError(f"Year must be between {MIN_YEAR} and {MAX_YEAR}, got {v}.")
        return v

    @field_validator("phenotype")
    @classmethod
    def validate_phenotype(cls, v):
        if v is None:
            return "unknown"
        if v not in VALID_PHENOTYPE:
            raise ValueError(f"Unknown phenotype '{v}'. Must be one of: {VALID_PHENOTYPE}")
        return v

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v):
        if v not in VALID_DATASET:
            raise ValueError(f"Unknown dataset '{v}'. Must be one of: {VALID_DATASET}")
        return v


# ------------------------
# Recommendation Endpoint
# ------------------------
@app.post("/recommend")
def recommend_antibiotics(patient: PatientInput):

    patient_dict = patient.model_dump()

    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_dict])

    # Generate ranked recommendations
    df = recommender.recommend(patient_df)

    # Apply clinical rules
    df = rules.apply(df)

    return {
        "recommendations": df.to_dict(orient="records")
    }