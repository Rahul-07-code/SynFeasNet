/**
 * types.ts — SynFeasNet Frontend Type Definitions
 *
 * These types map directly to the FastAPI PredictionResponse schema,
 * which in turn maps to the actual predict() output from inference/predict.py.
 *
 * No Gemini-hallucinated fields. Every field comes from real model output.
 */

// ═══════════════════════════════════════════════════════════════════════════
// CHEMISTRY
// ═══════════════════════════════════════════════════════════════════════════

export interface ChemistryProperties {
  molecular_weight: number;
  num_heavy_atoms: number;
  max_ring_size: number;
  is_macrocycle: boolean;
  num_rings: number;
  num_stereocenters: number;
  num_rotatable_bonds: number;
  logp: number;
  tpsa: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// SPI DIMENSIONS
// ═══════════════════════════════════════════════════════════════════════════

export interface SPIDimensions {
  synthetic_complexity: number;
  route_practicality: number;
  precursor_availability: number;
  scalability: number;
  retro_confidence: number;
  medchem_realism: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// RETROSYNTHESIS
// ═══════════════════════════════════════════════════════════════════════════

export interface RetroVisualizationNode {
  id: string;
  smiles: string;
  label: string;
  depth: number;
  type: "building_block" | "intermediate";
  is_building_block: boolean;
  is_leaf: boolean;
}

export interface RetroVisualizationEdge {
  source: string;
  target: string;
  reaction: string | null;
}

export interface RetroVisualization {
  nodes: RetroVisualizationNode[];
  edges: RetroVisualizationEdge[];
  layout: string;
  root_id: string | null;
}

export interface RetroRouteTreeNode {
  smiles: string;
  depth: number;
  reaction: string | null;
  is_building_block: boolean;
  is_leaf: boolean;
  children: RetroRouteTreeNode[];
}

export interface RetroRouteSummary {
  best_score: number | null;
  best_solved_fraction: number;
  best_n_steps: number;
}

export interface RetroRoute {
  rank: number;
  score: number;
  solved_fraction: number;
  n_steps: number;
  tree: RetroRouteTreeNode;
  visualization: RetroVisualization;
}

export interface RetrosynthesisResult {
  enabled: boolean;
  target_smiles: string;
  status: "ok" | "no_route" | "error";
  n_routes: number;
  summary: RetroRouteSummary;
  routes: RetroRoute[];
  visualization: RetroVisualization;
  scoring_inputs?: Record<string, number>;
  message?: string;
  error?: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// PREDICTION RESULT — the top-level response from POST /api/predict
// ═══════════════════════════════════════════════════════════════════════════

export interface PredictionResult {
  smiles: string;
  stage1_pass: boolean;
  stage1_prob: number;
  spi_score: number;
  spi_class: number;         // 0=intractable, 1=difficult, 2=challenging, 3=practical, 4=trivial
  spi_label: string;
  spi_dimensions: SPIDimensions;
  spi_report: string;
  chemistry: ChemistryProperties;
  warning: string;
  retrosynthesis?: RetrosynthesisResult;
}

// ═══════════════════════════════════════════════════════════════════════════
// API METADATA
// ═══════════════════════════════════════════════════════════════════════════

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  device: string;
}

export interface SystemStatus {
  model_loaded: boolean;
  checkpoint_path: string;
  checkpoint_exists: boolean;
  device: string;
  spi_dimensions: string[];
  retrosynthesis_engine_loaded: boolean;
}

export interface ModelMetrics {
  parameter_counts: Record<string, number> | null;
  checkpoint_info: Record<string, unknown> | null;
  spi_dimensions: string[];
  spi_class_labels: string[];
  spi_weights: Record<string, number>;
}

// ═══════════════════════════════════════════════════════════════════════════
// UI HELPERS
// ═══════════════════════════════════════════════════════════════════════════

export const SPI_CLASS_COLORS: Record<number, { bg: string; text: string; border: string; label: string }> = {
  0: { bg: "bg-red-50", text: "text-red-700", border: "border-red-200", label: "Intractable" },
  1: { bg: "bg-orange-50", text: "text-orange-700", border: "border-orange-200", label: "Difficult" },
  2: { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-200", label: "Challenging" },
  3: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200", label: "Practical" },
  4: { bg: "bg-blue-50", text: "text-blue-700", border: "border-blue-200", label: "Trivial" },
};

export const SPI_DIMENSION_LABELS: Record<string, string> = {
  synthetic_complexity: "Synthetic Complexity",
  route_practicality: "Route Practicality",
  precursor_availability: "Precursor Availability",
  scalability: "Scalability",
  retro_confidence: "Retro Confidence",
  medchem_realism: "MedChem Realism",
};
