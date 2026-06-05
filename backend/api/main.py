"""
main.py — SynFeasNet FastAPI Backend
======================================
Production-ready API server that wraps the real SynFeasNet inference pipeline.

Every endpoint calls actual model inference — no mocks, no fakes, no placeholders.

Endpoints:
    POST /api/predict          — SPI prediction + retrosynthesis for a SMILES string
    POST /api/retrosynthesis   — Standalone retrosynthesis analysis
    POST /api/predict/batch    — Batch SPI prediction
    GET  /api/metrics          — Model metadata and parameter counts
    GET  /api/health           — Health check
    GET  /api/status           — System status
"""

import os
import sys
import time
import traceback
from contextlib import asynccontextmanager

# ── Project root setup ────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    PredictRequest,
    RetrosynthesisRequest,
    BatchPredictRequest,
    PredictionResponse,
    BatchPredictionResponse,
    BatchPredictionItem,
    RetrosynthesisResult,
    HealthResponse,
    StatusResponse,
    ModelMetricsResponse,
    ErrorResponse,
    SPIDimensions,
    ChemistryProperties,
    Molecule3DRequest,
    Molecule3DResponse,
)

# ── Lazy imports from the actual SynFeasNet inference engine ──────────────────
# These are imported after sys.path is set up so they can find the models/ package
from inference.predict import (
    predict,
    predict_batch,
    predict_with_uncertainty,
    _ensure_loaded,
    _ensure_retro_loaded,
    _MODEL,
    _RETRO_ENGINE,
    CKPT_PATH,
    device as model_device,
)
from models.attention_fusion import SPI_DIMENSION_NAMES
from spi_labels import SPI_WEIGHTS, SPI_CLASS_THRESHOLDS
from retrosynthesis.engine import RetrosynthesisEngine
from rdkit import Chem
from rdkit.Chem import AllChem


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION LIFESPAN — model loading happens ONCE at startup
# ══════════════════════════════════════════════════════════════════════════════

_startup_time: float = 0.0
_model_loaded: bool = False
_retro_loaded: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and retrosynthesis engine at startup, exactly once."""
    global _startup_time, _model_loaded, _retro_loaded

    print("=" * 70)
    print("SynFeasNet FastAPI — Starting up...")
    print("=" * 70)

    t0 = time.time()

    # Load the SPI model + all extractors
    try:
        _ensure_loaded()
        _model_loaded = True
        print("[startup] ✓ SynPractIQ model loaded successfully")
    except Exception as e:
        print(f"[startup] ✗ FAILED to load model: {e}")
        print(traceback.format_exc())
        _model_loaded = False

    # Load retrosynthesis engine
    try:
        _ensure_retro_loaded()
        _retro_loaded = True
        print("[startup] ✓ Retrosynthesis engine loaded successfully")
    except Exception as e:
        print(f"[startup] ✗ FAILED to load retrosynthesis engine: {e}")
        _retro_loaded = False

    _startup_time = time.time() - t0
    print(f"[startup] Ready in {_startup_time:.1f}s")
    print("=" * 70)

    yield  # Application runs here

    print("[shutdown] SynFeasNet FastAPI shutting down.")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="SynFeasNet API",
    description="Production API for Synthetic Practicality Index prediction and retrosynthesis analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the frontend dev server
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL EXCEPTION HANDLER — never expose stack traces to clients
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    # Log the full traceback server-side
    print(f"[ERROR] Unhandled exception: {exc}")
    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Convert raw predict() dict → response schema
# ══════════════════════════════════════════════════════════════════════════════

def _build_prediction_response(smiles: str, result: dict) -> dict:
    """
    Transform the raw dict from predict() into the PredictionResponse schema.
    Handles both full results and partial results gracefully.
    """
    chemistry = result.get("chemistry", {})
    dimensions = result.get("spi_dimensions", {})
    retro_raw = result.get("retrosynthesis", None)

    response = {
        "smiles": smiles,
        "stage1_pass": result.get("stage1_pass", False),
        "stage1_prob": result.get("stage1_prob", 0.0),
        "spi_score": result.get("spi_score", 0.0),
        "spi_class": result.get("spi_class", 0),
        "spi_label": result.get("spi_label", "unknown"),
        "spi_dimensions": SPIDimensions(**dimensions) if dimensions else SPIDimensions(),
        "spi_report": result.get("spi_report", ""),
        "chemistry": ChemistryProperties(**chemistry) if chemistry else ChemistryProperties(),
        "warning": result.get("warning", ""),
    }

    if retro_raw is not None:
        response["retrosynthesis"] = retro_raw

    return response


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict(request: PredictRequest):
    """
    Predict the Synthetic Practicality Index for a SMILES string.

    Calls the actual SynFeasNet model — no mocks, no fakes.
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is starting up or checkpoint is missing.",
        )

    smiles = request.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=422, detail="SMILES string cannot be empty.")

    try:
        result = predict(
            smiles,
            include_retrosynthesis=request.include_retrosynthesis,
        )
        return _build_prediction_response(smiles, result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {e}")


@app.post("/api/retrosynthesis")
async def api_retrosynthesis(request: RetrosynthesisRequest):
    """
    Run standalone retrosynthesis analysis for a SMILES string.

    Uses the actual template-based retrosynthesis engine.
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is starting up or checkpoint is missing.",
        )

    smiles = request.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=422, detail="SMILES string cannot be empty.")

    try:
        # Run prediction first to get SPI scores needed for retro scoring
        result = predict(smiles, include_retrosynthesis=True)
        retro = result.get("retrosynthesis", {})

        return {
            "smiles": smiles,
            "spi_score": result.get("spi_score", 0.0),
            "retrosynthesis": retro,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {e}")


@app.post("/api/molecule/3d", response_model=Molecule3DResponse)
async def api_molecule_3d(request: Molecule3DRequest):
    """
    Generate 3D coordinates for a SMILES string using RDKit ETKDGv3.
    Returns the result as an SDF (Structure-Data File) string compatible with 3Dmol.js.
    """
    smiles = request.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=422, detail="SMILES string cannot be empty.")
        
    try:
        # Create RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Could not parse SMILES")
            
        # Add hydrogens for accurate 3D generation
        mol_h = Chem.AddHs(mol)
        
        # Generate conformer using ETKDG
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        
        conf_id = AllChem.EmbedMolecule(mol_h, params)
        if conf_id < 0:
            # Fallback
            conf_id = AllChem.EmbedMolecule(mol_h, randomSeed=42)
            if conf_id < 0:
                raise ValueError("Failed to generate 3D conformer")
                
        # Optimize geometry with MMFF
        try:
            res = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
            if res == -1:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
        except Exception:
            pass # Keep unoptimized coordinates if force field fails
            
        # Convert to Mol block (SDF format)
        sdf_block = Chem.MolToMolBlock(mol_h)
        
        return Molecule3DResponse(
            smiles=smiles,
            sdf_content=sdf_block
        )
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating 3D structure: {str(e)}")


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
async def api_predict_batch(request: BatchPredictRequest):
    """Batch prediction for multiple SMILES strings."""
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

    if not request.smiles_list:
        raise HTTPException(status_code=422, detail="smiles_list cannot be empty.")

    if len(request.smiles_list) > 100:
        raise HTTPException(
            status_code=422,
            detail="Batch size exceeds maximum of 100 molecules.",
        )

    raw_results = predict_batch(
        request.smiles_list,
        include_retrosynthesis=request.include_retrosynthesis,
    )

    items = []
    for r in raw_results:
        if "error" in r:
            items.append(BatchPredictionItem(smiles=r["smiles"], error=r["error"]))
        else:
            items.append(
                BatchPredictionItem(
                    **_build_prediction_response(r.get("smiles", ""), r)
                )
            )

    return BatchPredictionResponse(results=items)


@app.get("/api/health", response_model=HealthResponse)
async def api_health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if _model_loaded else "degraded",
        model_loaded=_model_loaded,
        device=str(model_device),
    )


@app.get("/api/status", response_model=StatusResponse)
async def api_status():
    """Detailed system status."""
    return StatusResponse(
        model_loaded=_model_loaded,
        checkpoint_path=CKPT_PATH,
        checkpoint_exists=os.path.exists(CKPT_PATH),
        device=str(model_device),
        spi_dimensions=list(SPI_DIMENSION_NAMES),
        retrosynthesis_engine_loaded=_retro_loaded,
    )


@app.get("/api/metrics", response_model=ModelMetricsResponse)
async def api_metrics():
    """Model metadata, parameter counts, and SPI configuration."""
    from inference.predict import _MODEL

    param_counts = None
    ckpt_info = None

    if _MODEL is not None:
        try:
            param_counts = _MODEL.count_parameters()
        except Exception:
            pass

    # Try to read checkpoint metadata
    if os.path.exists(CKPT_PATH):
        try:
            ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                ckpt_info = {
                    "epoch": ckpt.get("epoch", None),
                    "val_spi_mae": ckpt.get("val_spi_mae", None),
                    "val_spearman": ckpt.get("val_spearman", None),
                }
        except Exception:
            pass

    spi_class_labels = ["intractable", "difficult", "challenging", "practical", "trivial"]

    return ModelMetricsResponse(
        parameter_counts=param_counts,
        checkpoint_info=ckpt_info,
        spi_dimensions=list(SPI_DIMENSION_NAMES),
        spi_class_labels=spi_class_labels,
        spi_weights=dict(SPI_WEIGHTS),
    )


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))

    print(f"Starting SynFeasNet API on {host}:{port}")
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=False,  # Never reload in production — model is huge
    )
