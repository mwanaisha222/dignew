import pandas as pd

class AntibioticRecommender:

    def __init__(self, predictor, antibiotic_list):
        self.predictor = predictor
        self.antibiotics = antibiotic_list

    def recommend(self, patient_row):
        results = []

        for abx in self.antibiotics:
            temp = patient_row.copy()
            temp["antibiotic"] = abx

            prob_res = self.predictor.predict_resistance(temp)[0]

            results.append({
                "antibiotic": abx,
                "resistance_probability": prob_res,
                "success_probability": 1 - prob_res
            })

        df = pd.DataFrame(results)
        df = df.sort_values("success_probability", ascending=False)

        return df