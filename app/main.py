from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .model import EmotionModel

app = FastAPI(title="Live Speech Emotion Recognition")
model: EmotionModel | None = None

STATIC_DIR = Path(__file__).parent / "static"


@app.on_event("startup")
def _load_model() -> None:
    global model
    try:
        model = EmotionModel()
    except FileNotFoundError as exc:
        model = None
        print(f"[startup] {exc}")


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
    try:
        audio_bytes = await file.read()
        print(f"[predict] bytes={len(audio_bytes)} name={file.filename}")
        if not audio_bytes:
            return JSONResponse({"error": "Empty audio upload"}, status_code=400)
        if len(audio_bytes) < 1024:
            return JSONResponse({"error": "Audio clip too short"}, status_code=400)
        result = model.predict(audio_bytes)
        return JSONResponse(result)
    except Exception as exc:
        print(f"[predict] error: {exc}")
        return JSONResponse({"error": "Prediction failed"}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
