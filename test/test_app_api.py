import unittest
import requests

BASE_URL = "http://127.0.0.1:8000"


class TestHeartDiseaseAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Runs once before all tests"""
        cls.base_url = BASE_URL


    # Health Check Test
    def test_health_endpoint(self):
        response = requests.get(f"{self.base_url}/")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "API is operational")


    # Metrics Endpoint Test
    def test_metrics_endpoint(self):
        response = requests.get(f"{self.base_url}/metrics")
        self.assertEqual(response.status_code, 200)

        content = response.text
        self.assertIn("total_prediction_requests", content)
        self.assertIn("average_prediction_latency_seconds", content)


    # Prediction Endpoint Test
    def test_prediction_endpoint(self):
        payload = {
            "age": 55,
            "sex": 1,
            "chest_pain": 2,
            "resting_bp": 140,
            "chol": 220,
            "fasting_bs": 0,
            "rest_ecg": 1,
            "max_hr": 150,
            "exercise_angina": 0,
            "oldpeak": 2.3,
            "st_slope": 2,
            "Ca": 0,
            "thal": 2
        }

        response = requests.post(
            f"{self.base_url}/predict",
            json=payload
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("predicted_class", data)
        self.assertIn("probability", data)
        self.assertIn("latency_seconds", data)

        self.assertGreaterEqual(data["probability"], 0)
        self.assertLessEqual(data["probability"], 1)


    # Invalid Input Test
    def test_prediction_invalid_payload(self):
        payload = {
            "sex": 1,
            "chest_pain": 2,
            "resting_bp": 140,
            "chol": 220,
            "fasting_bs": 0,
            "rest_ecg": 1,
            "max_hr": 150,
            "exercise_angina": 0,
            "oldpeak": 2.3,
            "st_slope": 2,
            "Ca": 0,
            "thal": 2
        }  # Missing 'age'

        response = requests.post(
            f"{self.base_url}/predict",
            json=payload
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
