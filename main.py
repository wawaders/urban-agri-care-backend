
import io
import json
from pathlib import Path
import os
from openai import OpenAI
from pydantic import BaseModel

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from services import (
    DiagnosisUnavailableError,
    InvalidPredictionError,
    LowQualityImageError,
    diagnose_image,
)


BASE_DIR = Path(__file__).resolve().parent
TREATMENTS_FILE = BASE_DIR / "treatments.json"

app = FastAPI(title="Urban Agri-Care API")

# Development configuration.
# Restrict origins before production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def load_treatments():
    with open(TREATMENTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Urban Agri-Care API"
    }


@app.get("/treatments")
def list_treatments():
    treatments = load_treatments()

    return {
        "count": len(treatments),
        "treatments": [
            {
                "label": label,
                "crop": record.get("crop"),
                "disease": record.get("disease")
            }
            for label, record in treatments.items()
        ]
    }


@app.get("/treatments/{label}")
def get_treatment(label: str):
    treatments = load_treatments()
    treatment = treatments.get(label)

    if treatment is None:
        raise HTTPException(
            status_code=404,
            detail="No treatment record found for this label."
        )

    return {
        "label": label,
        "treatment": treatment
    }


@app.post("/diagnose")
async def diagnose_leaf(
    file: UploadFile = File(...)
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG or WebP image."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty."
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller."
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError
    ):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        ) from None

    try:
        diagnosis = diagnose_image(image_bytes)
    except LowQualityImageError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error
    except DiagnosisUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        ) from error
    except InvalidPredictionError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        ) from error
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The image could not be processed."
        ) from None

    label = diagnosis["label"]

    treatments = load_treatments()
    treatment = treatments.get(label)

    return {
        "success": True,
        "label": label,
        "confidence": diagnosis["confidence"],
        "model_version": diagnosis["model_version"],
        "treatment": treatment,
        "treatment_available": treatment is not None,
        "message": (
            "Diagnosis completed."
            if treatment is not None
            else "No treatment record is available for this label."
        )
    }
    
    # Sujay's Generative AI Integration
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class AIRequest(BaseModel):
    disease_name: str

@app.post("/ask-ai")
async def get_advanced_ai_treatment(request: AIRequest):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are an expert botanist and agricultural advisor. Provide highly detailed, structured recovery plans for plant diseases. Use short paragraphs and concise bullet points optimized for readability on a mobile phone screen. Do not use wide data tables."},
                {"role": "user", "content": f"An urban farmer uploaded a plant diagnosed with {request.disease_name}. Provide a 3-step advanced organic treatment."}
            ],
            temperature=0.7,
            max_tokens=2500
        )
        
        return {
            "success": True,
            "disease_analyzed": request.disease_name,
            "ai_solution": response.choices[0].message.content
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI API Error: {str(e)}")

