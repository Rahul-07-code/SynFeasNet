"""
schemas.py — Pydantic request/response models for the SynFeasNet FastAPI backend.

These models map directly to the output of inference/predict.py.
No fake data, no hallucinated fields — every field comes from actual model output.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule to predict")
    include_retrosynthesis: bool = Field(
        True, description="Whether to include retrosynthesis analysis"
    )


class RetrosynthesisRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the target molecule")


class Molecule3DRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule for 3D coordinate generation")


class BatchPredictRequest(BaseModel):
    smiles_list: List[str] = Field(
        ..., description="List of SMILES strings to predict"
    )
    include_retrosynthesis: bool = Field(
        True, description="Whether to include retrosynthesis analysis"
    )


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS — mapped from predict() output dict
# ══════════════════════════════════════════════════════════════════════════════

class ChemistryProperties(BaseModel):
    molecular_weight: float = 0.0
    num_heavy_atoms: int = 0
    max_ring_size: int = 0
    is_macrocycle: bool = False
    num_rings: int = 0
    num_stereocenters: int = 0
    num_rotatable_bonds: int = 0
    logp: float = 0.0
    tpsa: float = 0.0


class SPIDimensions(BaseModel):
    synthetic_complexity: float = 0.0
    route_practicality: float = 0.0
    precursor_availability: float = 0.0
    scalability: float = 0.0
    retro_confidence: float = 0.0
    medchem_realism: float = 0.0


class RetroVisualizationNode(BaseModel):
    id: str
    smiles: str
    label: str
    depth: int
    type: str  # "building_block" or "intermediate"
    is_building_block: bool
    is_leaf: bool


class RetroVisualizationEdge(BaseModel):
    source: str
    target: str
    reaction: Optional[str] = None


class RetroVisualization(BaseModel):
    nodes: List[RetroVisualizationNode] = []
    edges: List[RetroVisualizationEdge] = []
    layout: str = "tree"
    root_id: Optional[str] = None


class RetroRouteTreeNode(BaseModel):
    smiles: str
    depth: int
    reaction: Optional[str] = None
    is_building_block: bool = False
    is_leaf: bool = True
    children: List["RetroRouteTreeNode"] = []


class RetroRouteSummary(BaseModel):
    best_score: Optional[float] = None
    best_solved_fraction: float = 0.0
    best_n_steps: int = 0


class RetroRoute(BaseModel):
    rank: int
    score: float
    solved_fraction: float
    n_steps: int
    tree: RetroRouteTreeNode
    visualization: RetroVisualization


class RetrosynthesisResult(BaseModel):
    enabled: bool = True
    target_smiles: str = ""
    status: str = "no_route"  # "ok", "no_route", "error"
    n_routes: int = 0
    summary: RetroRouteSummary = RetroRouteSummary()
    routes: List[RetroRoute] = []
    visualization: RetroVisualization = RetroVisualization()
    scoring_inputs: Optional[Dict[str, float]] = None
    message: Optional[str] = None
    error: Optional[str] = None


class PredictionResponse(BaseModel):
    smiles: str
    stage1_pass: bool
    stage1_prob: float
    spi_score: float
    spi_class: int
    spi_label: str
    spi_dimensions: SPIDimensions
    spi_report: str
    chemistry: ChemistryProperties
    warning: str = ""
    retrosynthesis: Optional[RetrosynthesisResult] = None


class BatchPredictionItem(BaseModel):
    smiles: str
    stage1_pass: Optional[bool] = None
    stage1_prob: Optional[float] = None
    spi_score: Optional[float] = None
    spi_class: Optional[int] = None
    spi_label: Optional[str] = None
    spi_dimensions: Optional[SPIDimensions] = None
    chemistry: Optional[ChemistryProperties] = None
    warning: Optional[str] = None
    retrosynthesis: Optional[RetrosynthesisResult] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    results: List[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class StatusResponse(BaseModel):
    model_loaded: bool
    checkpoint_path: str
    checkpoint_exists: bool
    device: str
    spi_dimensions: List[str]
    retrosynthesis_engine_loaded: bool


class Molecule3DResponse(BaseModel):
    smiles: str
    sdf_content: str
    error: Optional[str] = None


class ModelMetricsResponse(BaseModel):
    parameter_counts: Optional[Dict[str, int]] = None
    checkpoint_info: Optional[Dict[str, object]] = None
    spi_dimensions: List[str]
    spi_class_labels: List[str]
    spi_weights: Dict[str, float]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
