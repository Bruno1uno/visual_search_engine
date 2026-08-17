import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from src.inference import VisualSearchEngine

CHECKPOINT_PATH = "checkpoints/best_resnet34_proxy_anchor.pt"
RESNET_FAISS_PATH = "indices/resnet_cub200.faiss"
CLIP_FAISS_PATH = "indices/clip_cub200.faiss"
METADATA_PATH = "indices/id_to_metadata.json"
METRICS_PATH = "metrics/eval_results_proxy_anchor.json"
DATA_IMAGES_DIR = "data/CUB_200_2011/images"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifecycle.
    Ensures CUB-200 dataset images are downloaded and loads VisualSearchEngine.
    """
    try:
        from src.dataset import download_and_extract_cub200
        download_and_extract_cub200("data")
    except Exception as err:
        print(f"Warning: Dataset download check failed: {err}")

    try:
        app.state.search_engine = VisualSearchEngine(
            resnet_checkpoint_path=CHECKPOINT_PATH,
            resnet_faiss_path=RESNET_FAISS_PATH,
            clip_faiss_path=CLIP_FAISS_PATH,
            metadata_path=METADATA_PATH,
        )
        print("VisualSearchEngine successfully initialized and ready.")
    except Exception as err:
        print(f"Warning: VisualSearchEngine failed to load at startup: {err}")
        app.state.search_engine = None
    yield
    app.state.search_engine = None


app = FastAPI(
    title="Visual Search Engine REST API",
    description=(
        "REST API for Deep Metric Learning Image Similarity Search (ResNet34 256D) "
        "and Multi-Modal Text-to-Image Retrieval (OpenCLIP 512D)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure images directory exists and mount static files to serve raw CUB-200 images to the web client
images_path = Path(DATA_IMAGES_DIR)
images_path.mkdir(parents=True, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=str(images_path)), name="images")



# Pydantic Response / Request Models
class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    num_indexed_vectors: int = 0


class SearchItemResponse(BaseModel):
    id: int
    image_id: int
    rel_path: str
    abs_path: str
    class_id: int
    class_name: str
    score: float


class SearchResponse(BaseModel):
    query_type: str
    engine_type: str
    top_k: int
    count: int
    results: list[SearchItemResponse]


class TextSearchRequest(BaseModel):
    text_query: str = Field(..., description="Text query prompt (e.g. 'yellow bird with black wings')")
    top_k: int = Field(8, ge=1, le=100)


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Healthcheck endpoint to verify server status and search engine readiness."""
    engine = getattr(app.state, "search_engine", None)
    is_ready = engine is not None
    num_vectors = len(engine.id_to_metadata) if is_ready else 0
    return HealthResponse(
        status="ok" if is_ready else "degraded",
        engine_ready=is_ready,
        num_indexed_vectors=num_vectors,
    )


@app.post("/api/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    engine_type: str = Query("resnet", description="Search engine: 'resnet' or 'clip'"),
    top_k: int = Query(8, ge=1, le=100),
):
    """Accepts an uploaded query image and returns the Top-K most similar images."""
    engine: VisualSearchEngine | None = getattr(app.state, "search_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="VisualSearchEngine is not initialized or indices are missing.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image (e.g. image/jpeg, image/png).",
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        results = engine.search_by_image(image=image, engine_type=engine_type, top_k=top_k)

        return SearchResponse(
            query_type="image",
            engine_type=engine_type,
            top_k=top_k,
            count=len(results),
            results=[SearchItemResponse(**res) for res in results],
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing image search query: {err}",
        )


@app.post("/api/search/text", response_model=SearchResponse)
async def search_by_text(request: TextSearchRequest):
    """Accepts a text prompt and performs multi-modal Text-to-Image retrieval via OpenCLIP."""
    engine: VisualSearchEngine | None = getattr(app.state, "search_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="VisualSearchEngine is not initialized or indices are missing.",
        )

    try:
        results = engine.search_by_text(text_query=request.text_query, top_k=request.top_k)

        return SearchResponse(
            query_type="text",
            engine_type="clip",
            top_k=request.top_k,
            count=len(results),
            results=[SearchItemResponse(**res) for res in results],
        )
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing text search query: {err}",
        )


@app.get("/api/metrics")
async def get_metrics():
    """Returns precomputed zero-shot evaluation metrics."""
    metrics_path = Path(METRICS_PATH)
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics JSON file not found.",
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# Mount compiled React frontend to serve single-page application on root path '/'
frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

