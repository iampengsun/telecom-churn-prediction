import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from telecom_churn.modeling import load_model
from telecom_churn.schema import Prediction, PredictRequest
from telecom_churn.scoring import score_customers

logger = logging.getLogger("telecom_churn.api")
STATIC = Path(__file__).parent / "static"


def create_app(model_path: Path | None = None, api_key: str | None = None) -> FastAPI:
    path = model_path or Path(os.getenv("CHURN_MODEL_PATH", "artifacts"))
    expected_key = api_key if api_key is not None else os.getenv("CHURN_API_KEY", "")
    environment = os.getenv("CHURN_ENV", "development")
    if environment == "production" and not expected_key:
        raise ValueError("CHURN_API_KEY is required in production")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = None
        try:
            app.state.bundle = load_model(path)
            logger.info("model_loaded version=%s", app.state.bundle["metadata"]["model_version"])
        except (FileNotFoundError, ValueError, KeyError):
            logger.exception("model_load_failed")
        yield
        app.state.bundle = None

    app = FastAPI(title="Telecom Churn Prediction", version="0.1.0", lifespan=lifespan)

    def authenticate(x_api_key: str | None = Header(default=None)):
        if expected_key and not secrets.compare_digest(x_api_key or "", expected_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def get_bundle():
        bundle = getattr(app.state, "bundle", None)
        if bundle is None:
            raise HTTPException(status_code=503, detail="Model unavailable")
        return bundle

    @app.middleware("http")
    async def request_metrics(request: Request, call_next):
        request_id = uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(bundle=Depends(get_bundle)):
        return {"status": "ready", "model_version": bundle["metadata"]["model_version"]}

    @app.get("/api/v1/model", dependencies=[Depends(authenticate)])
    def model_info(bundle=Depends(get_bundle)):
        return bundle["metadata"]

    @app.post(
        "/api/v1/predict", response_model=list[Prediction], dependencies=[Depends(authenticate)]
    )
    def predict(payload: PredictRequest, bundle=Depends(get_bundle)):
        frame = pd.DataFrame([row.model_dump() for row in payload.customers])
        try:
            return score_customers(frame, bundle).to_dict("records")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
