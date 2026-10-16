"""Request / response models for the prediction API."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """One patient record using the UCI Cleveland column definitions."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6,
    }})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=40, le=260, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: Optional[float] = Field(None, ge=0, le=4, description="Major vessels coloured (0-3)")
    thal: Optional[float] = Field(None, description="Thalassemia: 3 normal, 6 fixed, 7 reversible")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease likely, 0 = unlikely")
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float
    model_name: str
    request_id: str
