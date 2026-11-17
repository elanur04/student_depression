import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from streamlit.testing.v1 import AppTest
from inference import FEATURES, load_artifacts, predict_label, preprocess_input

SAMPLE = dict(age=25, academic_pressure=3, cgpa=7.0, study_satisfaction=3,
              diet="Sağlıksız", thoughts="Hayır", hours=8, financial_stress=3,
              family="Var", gender="Erkek", sleep="7-8 saat", education="Yüksek Lisans")

class InferenceTests(unittest.TestCase):
    def test_notebook_feature_order_and_categories(self):
        frame = preprocess_input(SAMPLE)
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertEqual(frame.iloc[0].tolist(), [25, 3, 7, 3, -1, 0, 8, 3, 1, 1, 2, 2])

    def test_unsupported_category_is_not_silently_mapped(self):
        with self.assertRaises(KeyError):
            preprocess_input(dict(SAMPLE, gender="Diğer"))

    def test_inference_uses_scaler_and_saved_model(self):
        transformed = np.zeros((1, 12))
        scaler, model = Mock(), Mock()
        scaler.transform.return_value = transformed
        model.predict.return_value = np.array([1])
        self.assertEqual(predict_label(SAMPLE, model, scaler), 1)
        scaler.transform.assert_called_once()
        model.predict.assert_called_once_with(transformed)

    def test_artifacts_load_outside_repository_and_predict(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                model, scaler = load_artifacts()
                self.assertIn(predict_label(SAMPLE, model, scaler), [0, 1])
            finally:
                os.chdir(original)

    def test_form_submits_a_real_prediction(self):
        app = AppTest.from_file(str(Path(__file__).with_name("main.py"))).run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Tahmin edilen veri seti etiketi" in item.value for item in app.markdown))

if __name__ == "__main__":
    unittest.main()
