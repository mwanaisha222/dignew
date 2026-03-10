import pandas as pd

class AntibioticRecommender:

    def __init__(self, predictor, antibiotic_list, species_antibiotic_map=None):
        self.predictor = predictor
        self.antibiotics = antibiotic_list
        self.species_antibiotic_map = species_antibiotic_map or {}

    def recommend(self, patient_row, species=None):
        results = []

        # Filter antibiotics to only those valid for this species
        if species and species in self.species_antibiotic_map:
            valid_set = set(self.species_antibiotic_map[species])
            valid_abx = [a for a in self.antibiotics if a in valid_set]
        else:
            valid_abx = self.antibiotics

        for abx in valid_abx:
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