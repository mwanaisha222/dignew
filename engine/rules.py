class ClinicalRules:

    def __init__(self, reserve_drugs):
        self.reserve_drugs = reserve_drugs

    def apply(self, df):
        df["clinical_score"] = df["success_probability"]

        # Penalize reserve drugs
        df.loc[df["antibiotic"].isin(self.reserve_drugs),
               "clinical_score"] -= 0.15

        return df.sort_values("clinical_score", ascending=False)