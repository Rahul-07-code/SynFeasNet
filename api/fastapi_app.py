"""
SynFeasNet v2 — FastAPI Application
======================================
Production API for synthetic feasibility prediction.

Endpoints:
  POST /predict   → feasibility, confidence, explanation, retrosynthesis
  GET  /health    → API + model health check
  POST /batch     → batch prediction (list of SMILES)

Run:
  python -m uvicorn api.fastapi_app:app --host 0.0.0.0 --port 8000

Or:
  cd SynFeasNet
  python api/fastapi_app.py
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from inference.predict import predict, predict_batch, _ensure_loaded
from retrosynthesis.providers import RetrosynthesisRouter
from explainability.explainer import SynFeasExplainer


# ══════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    smiles: str = Field(
        ..., description="SMILES string of the target molecule",
        json_schema_extra={"examples": ["CC(=O)Oc1ccccc1C(=O)O"]},
    )
    include_retrosynthesis: bool = Field(
        default=True, description="Include retrosynthetic analysis"
    )
    include_explanation: bool = Field(
        default=True, description="Include human-readable explanation"
    )


class BatchRequest(BaseModel):
    smiles_list: List[str] = Field(
        ..., description="List of SMILES strings",
        json_schema_extra={"examples": [["CCO", "c1ccccc1"]]},
    )


class RetroStepSchema(BaseModel):
    reactants: List[str] = []
    product: str = ""
    reaction_smiles: str = ""
    confidence: float = 0.0
    reaction_name: str = ""


class RetroSchema(BaseModel):
    target: str = ""
    steps: List[RetroStepSchema] = []
    num_steps: int = 0
    provider: str = ""
    success: bool = False
    message: str = ""


class PredictResponse(BaseModel):
    smiles: str
    probability: float = Field(ge=0.0, le=1.0)
    label: str
    confidence: str
    threshold: float
    chemistry: dict = {}
    warning: str = ""
    explanation: Optional[str] = None
    retrosynthesis: Optional[RetroSchema] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    device: str = "cpu"
    version: str = "2.0.0"


# ══════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════

_retro_router: RetrosynthesisRouter = None
_explainer: SynFeasExplainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and services on startup."""
    global _retro_router, _explainer

    print("[API] Starting SynFeasNet v2...")

    # Load the prediction model
    _ensure_loaded()
    print("[API] Model loaded")

    # Initialize retrosynthesis
    _retro_router = RetrosynthesisRouter()

    # Initialize explainer
    _explainer = SynFeasExplainer()

    yield

    print("[API] Shutting down")


# ══════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="SynFeasNet v2",
    description=(
        "Multi-modal synthetic feasibility prediction API.\n\n"
        "Predicts how easy a molecule is to synthesize using "
        "ANN + GAT + ChemBERTa + EGNN branches with attention fusion.\n\n"
        "Includes retrosynthetic analysis and explainability."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API and model health."""
    from inference.predict import _MODEL
    device_str = "not loaded"
    if _MODEL is not None:
        device_str = str(next(_MODEL.parameters()).device)

    return HealthResponse(
        status="ok",
        model_loaded=_MODEL is not None,
        device=device_str,
        version="2.0.0",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_endpoint(request: PredictRequest):
    """
    Predict synthetic feasibility of a molecule.

    **Input**: SMILES string
    **Output**: probability, label, confidence, explanation, retrosynthesis plan
    """
    smiles = request.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="Empty SMILES string")

    # Run prediction
    try:
        result = predict(smiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Explanation
    explanation_text = None
    if request.include_explanation and _explainer:
        try:
            expl = _explainer.explain(
                smiles, result["probability"], result["threshold"]
            )
            explanation_text = expl["text"]
        except Exception:
            explanation_text = None

    # Retrosynthesis
    retro = None
    if request.include_retrosynthesis and _retro_router:
        try:
            retro_result = _retro_router.analyze(smiles, max_steps=5)
            rd = retro_result.to_dict()
            retro = RetroSchema(
                target=rd["target"],
                steps=[RetroStepSchema(**s) for s in rd["steps"]],
                num_steps=rd["num_steps"],
                provider=rd["provider"],
                success=rd["success"],
                message=rd.get("message", ""),
            )
        except Exception:
            pass

    return PredictResponse(
        smiles=smiles,
        probability=result["probability"],
        label=result["label"],
        confidence=result["confidence"],
        threshold=result["threshold"],
        chemistry=result.get("chemistry", {}),
        warning=result.get("warning", ""),
        explanation=explanation_text,
        retrosynthesis=retro,
    )


@app.post("/batch", tags=["Prediction"])
async def batch_predict(request: BatchRequest):
    """
    Batch prediction for multiple SMILES strings.
    Returns a list of prediction results.
    """
    if len(request.smiles_list) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 SMILES per batch"
        )
    results = predict_batch(request.smiles_list)
    return {"results": results, "count": len(results)}


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        reload=False,
    )
