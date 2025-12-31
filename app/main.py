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
    audio_bytes = await file.read()
    result = model.predict(audio_bytes)
    return JSONResponse(result)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
