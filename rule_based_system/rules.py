import collections
import collections.abc

if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence

try:
    from experta import Fact, KnowledgeEngine, Rule, P
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'experta'. Install it with: "
        "python -m pip install experta"
    ) from exc


class Patient(Fact):
    pass


class HeartDiseaseEngine(KnowledgeEngine):
    def __init__(self):
        super().__init__()
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

    @Rule(Patient(chol=P(lambda x: x > 240), age=P(lambda x: x > 50)))
    def rule_high_chol_old_age(self):
        self._flag("High", "High cholesterol (>240) with age >50")

    @Rule(Patient(trestbps=P(lambda x: x > 140), exang=1))
    def rule_high_bp_exang(self):
        self._flag("High", "High BP (>140) with exercise-induced angina")

    @Rule(Patient(thalach=P(lambda x: x > 150), exang=0, age=P(lambda x: x < 55)))
    def rule_good_exercise(self):
        self._flag("Low", "Good exercise capacity + no angina + young age")

    @Rule(Patient(chol=P(lambda x: x > 200), trestbps=P(lambda x: x > 130), fbs=1))
    def rule_multiple_risks(self):
        self._flag("High", "Multiple risks: High cholesterol + BP + fasting blood sugar")

    @Rule(Patient(cp=P(lambda x: x >= 2), oldpeak=P(lambda x: x > 1.5)))
    def rule_chest_pain_st_depression(self):
        self._flag("High", "Significant chest pain with ST depression >1.5")

    @Rule(Patient(ca=P(lambda x: x >= 2)))
    def rule_major_vessels(self):
        self._flag("High", "2+ major vessels colored by fluoroscopy")

    @Rule(Patient(thal=P(lambda x: x in [1, 2])))
    def rule_thalassemia_defect(self):
        self._flag("Moderate", "Thalassemia defect detected")

    @Rule(Patient(age=P(lambda x: x < 45), chol=P(lambda x: x < 200), trestbps=P(lambda x: x < 120), fbs=0))
    def rule_young_healthy(self):
        self._flag("Low", "Young age with all normal health indicators")

    @Rule(Patient(restecg=0, exang=0))
    def rule_normal_ecg(self):
        self._flag("Low", "Normal ECG with no exercise angina")

    @Rule(Patient(slope=2, oldpeak=P(lambda x: x > 2.0)))
    def rule_severe_slope_oldpeak(self):
        self._flag("High", "Flat slope with high ST depression (>2.0)")

    @Rule(Patient(sex=0, cp=P(lambda x: x >= 1)))
    def rule_female_chest_pain(self):
        self._flag("Moderate", "Female with chest pain symptoms")

    @Rule(Patient(sex=1, age=P(lambda x: x > 55), exang=1))
    def rule_male_age_angina(self):
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
    engine.reset()
    engine.declare(Patient(**patient_data))
    engine.run()
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
