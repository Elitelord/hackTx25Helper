from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from dotenv import load_dotenv

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import difflib
import logging

# --- Basic logging ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Load environment variables ---
load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ GENAI_API_KEY is not set. Add it to your .env or export it before running.")

client = genai.Client(api_key=api_key)

# --- Initialize FastAPI app ---
app = FastAPI()

# --- Load ML artifacts once ---
model = load_model("models/plan_predictor.keras")
encoder = joblib.load("models/pay_freq_encoder.joblib")
scaler = joblib.load("models/numeric_scaler.joblib")
label_encoder = joblib.load("models/plan_label_encoder.joblib")

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://carta-ecru.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API Endpoint ---
@app.post("/parse-document")
async def parse_document(file: UploadFile = File(...)):
    # Read PDF bytes
    pdf_bytes = await file.read()

    # Define a prompt to extract relevant fields
    prompt = """
    Extract the following fields from this document:
    - Pay Frequency
    - Gross Pay Per Period
    - Total Taxes Withheld Per Period
    - Total Deductions Per Period
    - Net Pay Per Period
    - Credit Score

    Return each field as a float type in JSON format, removing any non-numeric characters like '$'.
    Example output format:
    {
      "pay_frequency": "biweekly",
      "gross_pay_per_period": 2404.37,
      "total_taxes_withheld_per_period": 615.68,
      "total_deductions_per_period": 116.78,
      "net_pay_per_period": 1671.91,
      "credit_score": 720
    }
    """

    # Send PDF to Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            genai.types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            prompt
        ],
    )

    log.info("Parsed Data from Gemini: %s", response.text)

    # Extract JSON from Gemini output
    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    if not match:
        raise HTTPException(status_code=400, detail="No JSON found in Gemini output.")

    try:
        parsed_json = json.loads(match.group())
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Failed to parse JSON inside Gemini output.")

    # --- Normalize and map pay_frequency to encoder's expected categories ---
    raw_pay_freq = parsed_json.get("pay_frequency", "")
    if not raw_pay_freq:
        raise HTTPException(status_code=400, detail="Missing pay_frequency in parsed JSON.")

    normalized = raw_pay_freq.strip().lower()

    # Get encoder known categories (if available)
    try:
        known_categories = list(encoder.categories_[0])
    except Exception:
        # If encoder doesn't expose categories_, fail with helpful message
        log.error("Encoder has no categories_ attribute. Encoder repr: %s", repr(encoder))
        raise HTTPException(status_code=500, detail="Encoder not compatible with category inspection.")

    # Build a lowercase -> original mapping for safe lookup
    lower_to_orig = {cat.lower(): cat for cat in known_categories}

    mapped_freq = None
    if normalized in lower_to_orig:
        mapped_freq = lower_to_orig[normalized]
    else:
        # Try fuzzy match on lowercase keys
        close = difflib.get_close_matches(normalized, list(lower_to_orig.keys()), n=1, cutoff=0.6)
        if close:
            mapped_freq = lower_to_orig[close[0]]

    if mapped_freq is None:
        # Return a 400 so frontend can show a user-friendly message instead of a 500
        raise HTTPException(
            status_code=400,
            detail=f"Unrecognized pay_frequency '{raw_pay_freq}'. Known values: {known_categories}"
        )

    # Build DataFrame in the model's expected format using the mapped frequency
    try:
        sample_df = pd.DataFrame([{
            "pay_frequency": mapped_freq,
            "gross_pay_per_period": float(parsed_json["gross_pay_per_period"]),
            "total_taxes_withheld_per_period": float(parsed_json["total_taxes_withheld_per_period"]),
            "total_deductions_per_period": float(parsed_json["total_deductions_per_period"]),
            "net_pay_per_period": float(parsed_json["net_pay_per_period"]),
            "credit_score": float(parsed_json["credit_score"]),
        }])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field in parsed JSON: {e}")

    # Preprocess
    pay_freq_enc = encoder.transform(sample_df[["pay_frequency"]])
    numeric_scaled = scaler.transform(sample_df[[
        "gross_pay_per_period",
        "total_taxes_withheld_per_period",
        "total_deductions_per_period",
        "net_pay_per_period",
        "credit_score",
    ]])
    X_sample = np.hstack([pay_freq_enc, numeric_scaled])

    # ✅ Predict probabilities
    pred_probs = model.predict(X_sample)[0]  # shape: (n_classes,)
    top3_idx = np.argsort(pred_probs)[-3:][::-1]

    # Map indices to labels + probabilities
    top3_plans = [
        {
            "plan": label_encoder.inverse_transform([i])[0],
            "probability": float(pred_probs[i]),
        }
        for i in top3_idx
    ]

    log.info("Top 3 Financial Plans For You: %s", top3_plans)

    # Return parsed data + top 3 plans
    return {
        "parsed_data": parsed_json,
        "top_3_predictions": top3_plans,
    }
