/**
 * RetrosynthesisView.tsx — Retrosynthesis Route Tree Visualization
 *
 * Renders actual retrosynthesis data from the SynFeasNet template-based engine.
 * Displays the route tree using real nodes/edges from the visualization JSON.
 *
 * No fake reagent images, no hallucinated literature precedents.
 * Every node is a real SMILES string from the retrosynthesis engine.
 */

import { useState, useMemo } from "react";
import { Download, ChevronDown, ChevronRight, Leaf, GitBranch, FlaskConical, Circle, Info } from "lucide-react";
import { PredictionResult, RetroVisualizationNode, RetroVisualizationEdge, RetroRoute, SPI_CLASS_COLORS } from "../types";

interface RetrosynthesisViewProps {
  predictionResult: PredictionResult | null;
}

// Recursive tree node component
function TreeNodeView({
  node,
  edges,
  allNodes,
  depth,
  expanded,
  onToggle,
}: {
  node: RetroVisualizationNode;
  edges: RetroVisualizationEdge[];
  allNodes: RetroVisualizationNode[];
  depth: number;
  expanded: Set<string>;
  onToggle: (id: string) => void;
}) {
  // Find children of this node
  const childEdges = edges.filter((e) => e.source === node.id);
  const childNodes = childEdges
    .map((e) => allNodes.find((n) => n.id === e.target))
    .filter(Boolean) as RetroVisualizationNode[];

  const isExpanded = expanded.has(node.id);
  const hasChildren = childNodes.length > 0;

  const nodeColor = node.is_building_block
    ? "border-emerald-400 bg-emerald-50/50"
    : depth === 0
    ? "border-blue-400 bg-blue-50/50"
    : "border-slate-300 bg-white";

  const labelColor = node.is_building_block
    ? "text-emerald-700"
    : depth === 0
    ? "text-blue-700"
    : "text-slate-700";

  return (
    <div className="flex flex-col items-center">
      {/* Node card */}
      <div
        className={`relative w-64 border-2 rounded-xl shadow-xs overflow-hidden transition-all hover:shadow-md ${nodeColor}`}
      >
        {/* Header */}
        <div
          className={`px-3 py-2 border-b border-slate-100 flex justify-between items-center cursor-pointer select-none ${
            hasChildren ? "hover:bg-slate-50/80" : ""
          }`}
          onClick={() => hasChildren && onToggle(node.id)}
        >
          <div className="flex items-center gap-2">
            {node.is_building_block ? (
              <Leaf size={12} className="text-emerald-500" />
            ) : depth === 0 ? (
              <FlaskConical size={12} className="text-blue-500" />
            ) : (
              <GitBranch size={12} className="text-slate-400" />
            )}
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
              {node.is_building_block
                ? "Building Block"
                : depth === 0
                ? "Target"
                : `Intermediate (d=${node.depth})`}
            </span>
          </div>
          {hasChildren && (
            <span className="text-slate-400">
              {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </span>
          )}
        </div>

        {/* SMILES */}
        <div className="p-3">
          <p
            className={`text-xs font-mono break-all leading-relaxed ${labelColor}`}
            title={node.smiles}
          >
            {node.smiles.length > 60 ? node.smiles.substring(0, 60) + "..." : node.smiles}
          </p>
        </div>

        {/* Status footer */}
        <div className="px-3 py-1.5 bg-slate-50/80 border-t border-slate-100 flex justify-between items-center text-[10px]">
          <span className="text-slate-400 font-semibold">Depth {node.depth}</span>
          <span
            className={`font-bold px-1.5 py-0.5 rounded text-[9px] ${
              node.is_building_block
                ? "bg-emerald-100 text-emerald-700"
                : node.is_leaf
                ? "bg-amber-100 text-amber-700"
                : "bg-slate-100 text-slate-600"
            }`}
          >
            {node.is_building_block ? "✓ Available" : node.is_leaf ? "Leaf" : "Expandable"}
          </span>
        </div>
      </div>

      {/* Children with connectors */}
      {hasChildren && isExpanded && (
        <div className="flex flex-col items-center mt-2">
          {/* Vertical connector from parent */}
          <div className="w-px h-6 bg-slate-300" />

          {/* Reaction labels on edges */}
          {childEdges.length > 0 && childEdges[0].reaction && (
            <div className="bg-white border border-slate-200 rounded-full px-3 py-1 shadow-xs text-[10px] font-semibold text-slate-600 my-1">
              {childEdges[0].reaction.replace(/_/g, " ")}
            </div>
          )}

          {/* Horizontal connector bar */}
          {childNodes.length > 1 && (
            <div className="flex items-start relative">
              <div
                className="absolute top-0 bg-slate-300"
                style={{
                  height: "1px",
                  left: "50%",
                  right: "50%",
                  transform: `scaleX(${childNodes.length})`,
                }}
              />
            </div>
          )}

          {/* Child nodes */}
          <div className="flex gap-6 mt-2 flex-wrap justify-center">
            {childNodes.map((child) => (
              <div key={child.id} className="flex flex-col items-center">
                <div className="w-px h-4 bg-slate-300" />
                <TreeNodeView
                  node={child}
                  edges={edges}
                  allNodes={allNodes}
                  depth={depth + 1}
                  expanded={expanded}
                  onToggle={onToggle}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function RetrosynthesisView({ predictionResult }: RetrosynthesisViewProps) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set(["0"]));
  const [selectedRouteIdx, setSelectedRouteIdx] = useState(0);

  const retro = predictionResult?.retrosynthesis;
  const routes = retro?.routes || [];
  const selectedRoute = routes[selectedRouteIdx];

  // Use the selected route's visualization, or the top-level one
  const viz = selectedRoute?.visualization || retro?.visualization;
  const nodes = viz?.nodes || [];
  const edges = viz?.edges || [];
  const rootNode = nodes.find((n) => n.id === (viz?.root_id || "0"));

  const toggleNode = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const expandAll = () => {
    setExpanded(new Set(nodes.map((n) => n.id)));
  };

  const collapseAll = () => {
    setExpanded(new Set(["0"]));
  };

  if (!predictionResult) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center gap-4 py-20">
        <GitBranch size={40} className="text-slate-300" />
        <div>
          <h3 className="font-bold text-slate-800 text-base">No Retrosynthesis Data</h3>
          <p className="text-slate-400 text-xs mt-1 max-w-sm mx-auto">
            Run a prediction on the Molecule Analysis page first. Retrosynthesis routes will appear here.
          </p>
        </div>
      </div>
    );
  }

  if (!retro || retro.status === "no_route" || nodes.length === 0) {
    return (
      <div className="flex flex-col gap-6 w-full animate-fade-in">
        <div className="flex justify-between items-end border-b border-slate-100 pb-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight font-headline">Retrosynthesis</h1>
            <p className="text-sm text-slate-500 mt-1">Template-based retrosynthetic route analysis.</p>
          </div>
        </div>

        <div className="bg-amber-50 border border-amber-200 rounded-xl p-6 flex items-start gap-3">
          <Info size={18} className="text-amber-600 mt-0.5 shrink-0" />
          <div>
            <h3 className="font-bold text-amber-800 text-sm">No Routes Found</h3>
            <p className="text-amber-700 text-xs mt-1">
              {retro?.message ||
                "No retrosynthesis route could be generated. The molecule may be a macrocycle, already a building block, or outside the current template coverage."}
            </p>
            <p className="text-amber-600 text-[10px] mt-2 font-mono">
              Target: {retro?.target_smiles || predictionResult.smiles}
            </p>
          </div>
        </div>
      </div>
    );
  }

  const summary = retro.summary;

  return (
    <div className="flex flex-col h-[calc(100vh-100px)] w-full text-slate-800 animate-fade-in">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-6 py-4 flex flex-col md:flex-row justify-between items-start md:items-end flex-shrink-0 gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Target Molecule</span>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
              SPI_CLASS_COLORS[predictionResult.spi_class]?.bg || "bg-slate-100"
            } ${SPI_CLASS_COLORS[predictionResult.spi_class]?.text || "text-slate-600"}`}>
              SPI: {predictionResult.spi_score.toFixed(3)}
            </span>
          </div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight font-headline font-mono text-sm">
            {retro.target_smiles.length > 60
              ? retro.target_smiles.substring(0, 60) + "..."
              : retro.target_smiles}
          </h1>
        </div>

        <div className="flex items-center gap-6">
          {/* Route stats */}
          <div className="flex gap-4">
            <div className="flex flex-col items-end">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Routes</span>
              <span className="text-lg font-bold text-slate-800 font-headline">{retro.n_routes}</span>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Best Score</span>
              <span className="text-lg font-bold text-blue-600 font-headline">
                {summary.best_score !== null ? summary.best_score.toFixed(3) : "—"}
              </span>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-[10px] font-bold text-slate-400 uppercase">Steps</span>
              <span className="text-lg font-bold text-slate-800 font-headline">{summary.best_n_steps}</span>
            </div>
          </div>

          <div className="h-8 w-px bg-slate-200" />

          <div className="flex gap-2">
            <button
              onClick={expandAll}
              className="bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold h-9 px-3 border border-slate-200 rounded-lg transition-colors"
            >
              Expand All
            </button>
            <button
              onClick={collapseAll}
              className="bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold h-9 px-3 border border-slate-200 rounded-lg transition-colors"
            >
              Collapse
            </button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Route tree visualization */}
        <div className="flex-1 overflow-auto p-6 relative bg-slate-50/50">
          {/* Dot grid pattern */}
          <div
            className="absolute inset-0 pointer-events-none opacity-[0.03]"
            style={{
              backgroundImage: "radial-gradient(circle at 1.5px 1.5px, black 1.5px, transparent 0)",
              backgroundSize: "20px 20px",
            }}
          />

          <div className="flex justify-center py-6 z-10 relative">
            {rootNode && (
              <TreeNodeView
                node={rootNode}
                edges={edges}
                allNodes={nodes}
                depth={0}
                expanded={expanded}
                onToggle={toggleNode}
              />
            )}
          </div>
        </div>

        {/* Right details panel */}
        <aside className="w-full lg:w-80 bg-white border-t lg:border-t-0 lg:border-l border-slate-200 flex flex-col flex-shrink-0 shadow-xs">
          <div className="p-4 border-b border-slate-200 bg-slate-50/50">
            <h3 className="text-sm font-semibold text-slate-800 font-headline">Route Details</h3>
          </div>
          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-5">
            {/* Route selector */}
            {routes.length > 1 && (
              <div>
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                  Route Selection
                </h4>
                <div className="flex flex-col gap-1.5">
                  {routes.map((route, idx) => (
                    <button
                      key={idx}
                      onClick={() => setSelectedRouteIdx(idx)}
                      className={`text-left px-3 py-2 rounded-lg border text-xs font-medium transition-all ${
                        idx === selectedRouteIdx
                          ? "bg-blue-50 border-blue-200 text-blue-700"
                          : "bg-white border-slate-200 text-slate-600 hover:border-slate-300"
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <span>Route #{route.rank}</span>
                        <span className="font-mono font-bold">{route.score.toFixed(3)}</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Selected route KPIs */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-150">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide block mb-1">
                  Route Score
                </span>
                <span className="text-lg font-extrabold text-slate-850 font-headline">
                  {selectedRoute ? selectedRoute.score.toFixed(3) : summary.best_score?.toFixed(3) || "—"}
                </span>
              </div>
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-150">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide block mb-1">
                  Steps
                </span>
                <span className="text-lg font-extrabold text-slate-850 font-headline">
                  {selectedRoute ? selectedRoute.n_steps : summary.best_n_steps}
                </span>
              </div>
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-150 col-span-2">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide block mb-1">
                  Solved Fraction
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-base font-extrabold text-slate-850 font-headline">
                    {((selectedRoute ? selectedRoute.solved_fraction : summary.best_solved_fraction) * 100).toFixed(1)}%
                  </span>
                  <div className="flex-1 bg-slate-200 rounded-full h-1.5 overflow-hidden">
                    <div
                      className="bg-emerald-500 h-1.5 rounded-full"
                      style={{
                        width: `${(selectedRoute ? selectedRoute.solved_fraction : summary.best_solved_fraction) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Node statistics */}
            <div>
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                Node Statistics
              </h4>
              <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg flex flex-col gap-2 text-xs">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 font-medium">Total Nodes</span>
                  <span className="text-slate-800 font-bold">{nodes.length}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 font-medium">Building Blocks</span>
                  <span className="text-emerald-600 font-bold">
                    {nodes.filter((n) => n.is_building_block).length}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 font-medium">Intermediates</span>
                  <span className="text-slate-800 font-bold">
                    {nodes.filter((n) => !n.is_building_block).length}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 font-medium">Leaf Nodes</span>
                  <span className="text-slate-800 font-bold">
                    {nodes.filter((n) => n.is_leaf).length}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 font-medium">Reaction Edges</span>
                  <span className="text-slate-800 font-bold">{edges.length}</span>
                </div>
              </div>
            </div>

            {/* Scoring inputs */}
            {retro.scoring_inputs && (
              <div>
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                  Scoring Inputs
                </h4>
                <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg flex flex-col gap-2 text-xs font-mono">
                  {Object.entries(retro.scoring_inputs).map(([key, val]) => (
                    <div key={key} className="flex justify-between items-center">
                      <span className="text-slate-400">{key.replace(/_/g, " ")}</span>
                      <span className="text-slate-800 font-bold">{(val as number).toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
