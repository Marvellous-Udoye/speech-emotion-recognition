import json
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .model import EmotionModel

model: EmotionModel | None = None

STATIC_DIR = Path(__file__).parent / "static"
ROOT_DIR = Path(__file__).resolve().parents[1]
USER_RESULTS_DIR = ROOT_DIR / "results" / "users"


def _sanitize_device_id(device_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in device_id)


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
        device_id = request.headers.get("x-device-id", "unknown")
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
        try:
            safe_device_id = _sanitize_device_id(device_id) or "unknown"
            user_dir = USER_RESULTS_DIR / safe_device_id
            user_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "device_id": safe_device_id,
                "label": result.get("label"),
                "confidence": result.get("confidence"),
                "scores": result.get("scores"),
                "audio_bytes_len": len(audio_bytes),
                "content_type": content_type,
                "upload_name": request.headers.get("x-upload-name"),
                "upload_source": request.headers.get("x-upload-source"),
            }
            with (user_dir / "predictions.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as log_exc:
            print(f"[predict] log error: {log_exc}", flush=True)
        return JSONResponse(result)
    except Exception as exc:
        print(f"[predict] error: {exc}", flush=True)
        traceback.print_exc()
        return JSONResponse({"error": "Prediction failed"}, status_code=500)


@app.post("/api/session")
async def create_session(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON payload"}, status_code=400)

    device_id = (payload.get("device_id") or "").strip()
    profile = payload.get("profile")
    if not device_id:
        return JSONResponse({"error": "Missing device_id"}, status_code=400)
    if not isinstance(profile, dict):
        return JSONResponse({"error": "Missing profile"}, status_code=400)

    safe_device_id = "".join(
        ch for ch in device_id if ch.isalnum() or ch in ("-", "_")
    )
    if not safe_device_id:
        return JSONResponse({"error": "Invalid device_id"}, status_code=400)

    user_dir = USER_RESULTS_DIR / safe_device_id
    user_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    session_payload = {
        "device_id": safe_device_id,
        "profile": profile,
        "created_at": payload.get("created_at") or now,
        "updated_at": now,
        "user_agent": request.headers.get("user-agent"),
    }
    (user_dir / "session.json").write_text(
        json.dumps(session_payload, indent=2), encoding="utf-8"
    )
    return JSONResponse({"status": "ok", "device_id": safe_device_id})


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
