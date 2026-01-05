# Required Libraries
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field
import logging
import time


# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HeartAPI")


# Model & Scaler Paths
MODEL_FILE = "artifacts/model.pkl"
SCALER_FILE = "artifacts/scaler.pkl"

# Load trained artifacts with error handling
try:
    with open(MODEL_FILE, "rb") as mf:
        model = pickle.load(mf)
    with open(SCALER_FILE, "rb") as sf:
        scaler = pickle.load(sf)
except FileNotFoundError as e:
    logger.error(f"Model or scaler file not found: {e}")
    raise


# In-memory metrics tracking
TOTAL_REQUESTS = 0
TOTAL_TIME_SECONDS = 0.0


# Initialize FastAPI Application
app = FastAPI(title="Heart Disease Prediction API", version="1.1")


# Request Payload Schema with validation
class HeartInputSchema(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex: 0 = female, 1 = male")
    chest_pain: int
    resting_bp: int
    chol: int
    fasting_bs: int
    rest_ecg: int
    max_hr: int
    exercise_angina: int
    oldpeak: float
    st_slope: int
    Ca: int
    thal: int


# Health Check Endpoint
@app.get("/", tags=["Health"])
def api_health():
    """Simple health check to verify API status."""
    logger.info("Health check endpoint called")
    return {"status": "API is operational"}


# Prediction Endpoint
@app.post("/predict", tags=["Prediction"])
def predict(input_data: HeartInputSchema):
    """Predict heart disease risk for a given patient input."""
    global TOTAL_REQUESTS, TOTAL_TIME_SECONDS
    start_time = time.time()
    TOTAL_REQUESTS += 1

    try:
        logger.info(f"Received prediction request: {input_data.model_dump()}")

        # Convert input to numpy array
        features_array = np.array([[input_data.age, input_data.sex, input_data.chest_pain,
                                    input_data.resting_bp, input_data.chol, input_data.fasting_bs,
                                    input_data.rest_ecg, input_data.max_hr, input_data.exercise_angina,
                                    input_data.oldpeak, input_data.st_slope, input_data.Ca, input_data.thal]])
        
        # Scale features
        scaled_features = scaler.transform(features_array)

        # Predict class and probability
        pred_label = model.predict(scaled_features)[0]
        pred_prob = model.predict_proba(scaled_features)[0][1]

        # Track latency
        elapsed = time.time() - start_time
        TOTAL_TIME_SECONDS += elapsed

        logger.info(
            f"Prediction completed | Label: {pred_label} | Probability: {pred_prob:.4f} | "
            f"Time: {elapsed:.4f}s"
        )

        return JSONResponse(
            content={
                "predicted_class": int(pred_label),
                "probability": round(float(pred_prob), 4),
                "latency_seconds": round(elapsed, 4)
            }
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction could not be completed")


# Metrics Endpoint (Prometheus style)
@app.get("/metrics", response_class=PlainTextResponse, tags=["Metrics"])
def api_metrics():
    """Return API metrics for monitoring purposes."""
    average_latency = TOTAL_TIME_SECONDS / TOTAL_REQUESTS if TOTAL_REQUESTS else 0

    return f"""
        # HELP total_prediction_requests Total number of prediction requests
        # TYPE total_prediction_requests counter
        total_prediction_requests {TOTAL_REQUESTS}

        # HELP average_prediction_latency_seconds Average latency per request in seconds
        # TYPE average_prediction_latency_seconds gauge
        average_prediction_latency_seconds {average_latency:.4f}
        """


# Catch-all Exception Logging
@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred."}
    )
