"""Inference with the 12 features used by the saved notebook model."""
from pathlib import Path
import joblib
import pandas as pd

FEATURES = ["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Dietary Habits",
            "Have you ever had suicidal thoughts ?", "Work/Study Hours", "Financial Stress",
            "Family History of Mental Illness", "Gender_Male", "Sleep_Duration_UIQ", "Education_Level"]
DIET = {"Sağlıklı": 1, "Orta": 0, "Sağlıksız": -1, "Diğer": -2}
YES_NO = {"Hayır": 0, "Evet": 1}
FAMILY = {"Yok": 0, "Var": 1}
GENDER = {"Kadın": 0, "Erkek": 1}
SLEEP = {"5 saatten az": 0, "5-6 saat": 1, "7-8 saat": 2, "8 saatten fazla": 3}
# LabelEncoder sorted: Doctorate, Other, Postgraduate, Undergraduate.
EDUCATION = {"Doktora": 0, "Diğer / Lise": 1, "Yüksek Lisans": 2, "Lisans": 3}

def load_artifacts():
    directory = Path(__file__).resolve().parent
    model = joblib.load(directory / "model_svm.pkl")
    scaler = joblib.load(directory / "scaler.pkl")
    if list(scaler.feature_names_in_) != FEATURES or model.n_features_in_ != len(FEATURES):
        raise ValueError("Model ve ölçekleyici özellikleri uygulamayla uyumlu değil.")
    return model, scaler

def preprocess_input(data):
    values = [data["age"], data["academic_pressure"], data["cgpa"], data["study_satisfaction"],
              DIET[data["diet"]], YES_NO[data["thoughts"]], data["hours"], data["financial_stress"],
              FAMILY[data["family"]], GENDER[data["gender"]], SLEEP[data["sleep"]], EDUCATION[data["education"]]]
    return pd.DataFrame([values], columns=FEATURES, dtype=float)

def predict_label(data, model, scaler):
    scaled = scaler.transform(preprocess_input(data))
    return int(model.predict(scaled)[0])
