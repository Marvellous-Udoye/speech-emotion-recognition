import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .model import EmotionModel

model: EmotionModel | None = None

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = EmotionModel()
        print("[startup] model loaded", flush=True)
    except FileNotFoundError as exc:
        model = None
        print(f"[startup] {exc}", flush=True)

    yield
    print("[shutdown] app stopping", flush=True)


app = FastAPI(
    title="Live Speech Emotion Recognition",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    content_type = request.headers.get("content-type")
    content_length = request.headers.get("content-length")
    client = request.client
    client_addr = f"{client.host}:{client.port}" if client else "unknown"
    print(
        f"[request] {request.method} {request.url.path} ct={content_type} len={content_length} client={client_addr}",
        flush=True,
    )
    response = await call_next(request)
    elapsed_ms = int((time.time() - start) * 1000)
    print(
        f"[request] {request.method} {request.url.path} status={response.status_code} elapsed_ms={elapsed_ms}",
        flush=True,
    )
    return response


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(request: Request) -> JSONResponse:
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
    try:
        content_type = request.headers.get("content-type") or ""
        audio_bytes = b""
        name = request.headers.get("x-upload-name", "raw-body")
        if content_type.startswith("multipart/form-data"):
            print("[predict] parsing multipart form", flush=True)
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                return JSONResponse({"error": "Missing file field"}, status_code=400)
            audio_bytes = await upload.read()
            name = getattr(upload, "filename", "uploaded-file")
        else:
            print("[predict] reading raw body", flush=True)
            audio_bytes = await request.body()
        print(
            f"[predict] bytes={len(audio_bytes)} name={name} content_type={request.headers.get('content-type')}",
            flush=True,
        )
        if not audio_bytes:
            return JSONResponse({"error": "Empty audio upload"}, status_code=400)
        if len(audio_bytes) < 1024:
            return JSONResponse({"error": "Audio clip too short"}, status_code=400)
        result = model.predict(audio_bytes)
        return JSONResponse(result)
    except Exception as exc:
        print(f"[predict] error: {exc}", flush=True)
        traceback.print_exc()
        return JSONResponse({"error": "Prediction failed"}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
