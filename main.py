from fastapi import FastAPI, Header, HTTPException
from audio_utils import load_audio_from_base64
from feature_utils import extract_features
from predict import predict_voice
from config import API_KEY

app = FastAPI()

@app.post("/api/voice-detection")
def detect_voice(data: dict, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    if data.get("audioFormat", "").lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format allowed")

    if "audioBase64" not in data or "language" not in data:
        raise HTTPException(status_code=400, detail="Missing required fields")

    try:
        audio = load_audio_from_base64(data["audioBase64"])
        features = extract_features(audio)
        label, score, reason = predict_voice(features)

        return {
            "status": "success",
            "language": data["language"],
            "classification": label,
            "confidenceScore": round(float(score), 2),
            "explanation": reason
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
