class HeartDiseaseEngine:
    def __init__(self):
        self.matched_rules = []
        self.high_risk = False
        self.moderate_risk = False
        self.low_risk = False

    def _flag(self, level: str, message: str) -> None:
        self.matched_rules.append(message)
        if level == "High":
            self.high_risk = True
        elif level == "Moderate":
            self.moderate_risk = True
        elif level == "Low":
            self.low_risk = True

    def evaluate(self, patient: dict):
        if patient.get('chol', 0) > 240 and patient.get('age', 0) > 50:
            self._flag("High", "High cholesterol (>240) with age >50")
            
        if patient.get('trestbps', 0) > 140 and patient.get('exang', 0) == 1:
            self._flag("High", "High BP (>140) with exercise-induced angina")
            
        if patient.get('thalach', 0) > 150 and patient.get('exang', -1) == 0 and patient.get('age', 100) < 55:
            self._flag("Low", "Good exercise capacity + no angina + young age")
            
        if patient.get('chol', 0) > 200 and patient.get('trestbps', 0) > 130 and patient.get('fbs', 0) == 1:
            self._flag("High", "Multiple risks: High cholesterol + BP + fasting blood sugar")
            
        if patient.get('cp', 0) >= 2 and patient.get('oldpeak', 0) > 1.5:
            self._flag("High", "Significant chest pain with ST depression >1.5")
            
        if patient.get('ca', 0) >= 2:
            self._flag("High", "2+ major vessels colored by fluoroscopy")
            
        if patient.get('thal', -1) in [1, 2]:
            self._flag("Moderate", "Thalassemia defect detected")
            
        if patient.get('age', 100) < 45 and patient.get('chol', 1000) < 200 and patient.get('trestbps', 1000) < 120 and patient.get('fbs', -1) == 0:
            self._flag("Low", "Young age with all normal health indicators")
            
        if patient.get('restecg', -1) == 0 and patient.get('exang', -1) == 0:
            self._flag("Low", "Normal ECG with no exercise angina")
            
        if patient.get('slope', -1) == 2 and patient.get('oldpeak', 0) > 2.0:
            self._flag("High", "Flat slope with high ST depression (>2.0)")
            
        if patient.get('sex', -1) == 0 and patient.get('cp', 0) >= 1:
            self._flag("Moderate", "Female with chest pain symptoms")
            
        if patient.get('sex', -1) == 1 and patient.get('age', 0) > 55 and patient.get('exang', -1) == 1:
            self._flag("High", "Male >55 years with exercise-induced angina")

    def get_result(self) -> dict:
        if self.high_risk:
            final_risk = "High"
        elif self.moderate_risk:
            final_risk = "Moderate"
        elif self.low_risk:
            final_risk = "Low"
        else:
            final_risk = "Moderate"
            self.matched_rules.append("General assessment based on available data")

        return {
            "risk_level": final_risk,
            "matched_rules": self.matched_rules,
            "num_rules_matched": len(self.matched_rules),
        }

def assess_patient(patient_data: dict) -> dict:
    engine = HeartDiseaseEngine()
    engine.evaluate(patient_data)
    return engine.get_result()

if __name__ == "__main__":
    test_patient = {
        "age": 65,
        "sex": 1,
        "cp": 3,
        "trestbps": 150,
        "chol": 260,
        "fbs": 1,
        "restecg": 1,
        "thalach": 120,
        "exang": 1,
        "oldpeak": 2.5,
        "slope": 2,
        "ca": 3,
        "thal": 2,
    }
    print(assess_patient(test_patient))
