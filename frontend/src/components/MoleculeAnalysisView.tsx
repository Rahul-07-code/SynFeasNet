/**
 * MoleculeAnalysisView.tsx — SPI Prediction Results Display
 *
 * Displays actual SynFeasNet prediction output: SPI score, SPI dimensions,
 * stage 1 gate, chemistry properties, warnings, and the SPI report.
 *
 * Every data field comes from the real model — no mocks.
 */

import { useState } from "react";
import { Upload, Edit3, Sparkles, Loader2, Info, ShieldCheck, ShieldAlert, AlertTriangle, ChevronDown, ChevronRight } from "lucide-react";
import MoleculeViewer3D from "./MoleculeViewer3D";
import { PredictionResult, SPI_CLASS_COLORS, SPI_DIMENSION_LABELS } from "../types";

interface MoleculeAnalysisViewProps {
  predictionResult: PredictionResult | null;
  loading: boolean;
  onAnalyze: (smiles: string) => void;
  errorMsg: string | null;
}

export default function MoleculeAnalysisView({
  predictionResult,
  loading,
  onAnalyze,
  errorMsg
}: MoleculeAnalysisViewProps) {
  const [smilesInput, setSmilesInput] = useState(predictionResult?.smiles || "");
  const [showReport, setShowReport] = useState(false);

  const quickMolecules = [
    { name: "Ethanol", smiles: "CCO" },
    { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
    { name: "Caffeine", smiles: "Cn1cnc2c1c(=O)n(c(=O)n2C)C" },
    { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O" },
    { name: "Paracetamol", smiles: "CC(=O)Nc1ccc(O)cc1" },
  ];

  const handleQuickSelect = (smiles: string) => {
    setSmilesInput(smiles);
    onAnalyze(smiles);
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (smilesInput.trim()) {
      onAnalyze(smilesInput.trim());
    }
  };

  const data = predictionResult;
  const classStyle = data ? SPI_CLASS_COLORS[data.spi_class] || SPI_CLASS_COLORS[0] : null;

  return (
    <div className="flex flex-col gap-6 w-full animate-fade-in">
      {/* Page Header */}
      <div className="flex justify-between items-end border-b border-slate-100 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight font-headline">Molecule Analysis</h1>
          <p className="text-sm text-slate-500 mt-1">
            Enter a SMILES string to predict the Synthetic Practicality Index using the real SynFeasNet model.
          </p>
        </div>
      </div>

      {/* Input Section Card */}
      <div className="bg-white border border-slate-200/80 rounded-xl p-5 flex flex-col gap-4 shadow-xs">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-3 border-b border-slate-100 pb-3">
          <h3 className="text-sm font-semibold text-slate-800 flex items-center gap-1.5 font-headline">
            <span className="w-2 h-2 rounded-full bg-blue-500"></span> SMILES Input
          </h3>
        </div>

        {/* SMILES input form */}
        <form onSubmit={handleFormSubmit} className="flex flex-col md:flex-row items-end gap-3 pt-1">
          <div className="flex-grow flex flex-col gap-1.5 w-full">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wide">SMILES String</label>
            <input
              type="text"
              value={smilesInput}
              onChange={(e) => setSmilesInput(e.target.value)}
              placeholder="Enter SMILES e.g., CC(=O)Oc1ccccc1C(=O)O"
              className="w-full h-10 px-3.5 rounded-lg border border-slate-200 bg-slate-50/50 text-xs font-mono focus:border-blue-500 focus:bg-white focus:ring-1 focus:ring-blue-500 outline-hidden transition-all shadow-inner text-slate-800"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="h-10 px-5 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs rounded-lg flex items-center justify-center gap-2 transition-all disabled:opacity-75 tracking-wide shadow-sm shrink-0 w-full md:w-auto"
          >
            {loading ? (
              <>
                <Loader2 size={14} className="animate-spin" /> Predicting...
              </>
            ) : (
              <>Predict SPI</>
            )}
          </button>
        </form>

        {/* Quick presets */}
        <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-slate-100">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 mr-1.5 flex items-center gap-1">
            <Sparkles size={11} className="text-blue-500" /> Presets:
          </span>
          {quickMolecules.map((m, i) => (
            <button
              key={i}
              type="button"
              onClick={() => handleQuickSelect(m.smiles)}
              disabled={loading}
              className={`text-xs px-2.5 py-1 rounded-md border font-medium transition-all ${
                smilesInput === m.smiles
                  ? "bg-blue-50 border-blue-200 text-blue-700 font-semibold"
                  : "bg-slate-50 border-slate-200 text-slate-600 hover:border-slate-300"
              }`}
            >
              {m.name}
            </button>
          ))}
        </div>

        {errorMsg && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-xs py-2 px-3.5 rounded-md flex items-center gap-2 font-medium">
            <Info size={14} className="shrink-0 text-red-500" />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>

      {loading ? (
        /* Loading animation */
        <div className="bg-white border border-slate-100 rounded-2xl p-12 flex flex-col items-center justify-center text-center gap-4 py-20 shadow-xs border-dashed">
          <div className="relative flex items-center justify-center">
            <div className="w-12 h-12 rounded-full border-4 border-slate-100 border-t-blue-600 animate-spin" />
            <Sparkles size={20} className="text-blue-500 absolute animate-pulse" />
          </div>
          <div>
            <h3 className="font-bold text-slate-800 text-base">Running SynFeasNet Inference</h3>
            <p className="text-slate-400 text-xs mt-1 max-w-sm mx-auto">
              Loading model branches (ANN, GAT, ChemBERTa, EGNN), computing features, running attention fusion...
            </p>
          </div>
        </div>
      ) : data ? (
        /* Results Panel */
        <div className="grid grid-cols-12 gap-5 items-start">
          {/* Left Column: Molecule Viewer */}
          <div className="col-span-12 lg:col-span-8 bg-white border border-slate-200/80 rounded-xl overflow-hidden flex flex-col shadow-xs">
            <div className="px-4 py-3 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center z-10">
              <h3 className="text-xs font-semibold text-slate-800 uppercase tracking-wider font-headline">
                3D Molecule Structure
              </h3>
              <span className="text-xs font-mono font-medium text-blue-600 px-2 py-0.5 bg-blue-50 border border-blue-200/50 rounded">
                {data.smiles.length > 50 ? data.smiles.substring(0, 50) + "..." : data.smiles}
              </span>
            </div>
            <MoleculeViewer3D smiles={data.smiles} />
          </div>

          {/* Right Column: SPI Score Card */}
          <div className="col-span-12 lg:col-span-4 flex flex-col gap-5">
            {/* SPI Score */}
            <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
              <h3 className="text-[10px] uppercase font-bold tracking-wider text-slate-400 mb-2">
                Synthetic Practicality Index
              </h3>
              <div className="flex items-end gap-1.5 mb-3">
                <span className="text-4xl font-black font-headline text-blue-600 tracking-tight">
                  {data.spi_score.toFixed(3)}
                </span>
                <span className="text-xs font-semibold text-slate-400 mb-1">/ 1.0</span>
              </div>

              {/* SPI Class badge */}
              {classStyle && (
                <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-bold border ${classStyle.bg} ${classStyle.text} ${classStyle.border}`}>
                  {data.spi_class >= 3 ? <ShieldCheck size={14} /> : data.spi_class >= 2 ? <AlertTriangle size={14} /> : <ShieldAlert size={14} />}
                  Class {data.spi_class} — {classStyle.label}
                </div>
              )}

              {/* Stage 1 Gate */}
              <div className="mt-4 pt-3 border-t border-slate-100 space-y-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-500 font-medium">Stage 1 Gate</span>
                  <span className={`font-bold ${data.stage1_pass ? "text-emerald-600" : "text-red-600"}`}>
                    {data.stage1_pass ? "✓ PASS" : "✗ FAIL"} ({(data.stage1_prob * 100).toFixed(1)}%)
                  </span>
                </div>

                {data.warning && (
                  <div className="bg-amber-50 border border-amber-200 text-amber-700 text-[11px] py-1.5 px-2.5 rounded-md font-medium">
                    ⚠ {data.warning}
                  </div>
                )}
              </div>
            </div>

            {/* SPI Dimensions */}
            <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
              <h3 className="text-[10px] uppercase font-bold tracking-wider text-slate-400 mb-3">
                SPI Dimensions
              </h3>
              <div className="space-y-3">
                {Object.entries(data.spi_dimensions).map(([key, value]) => {
                  const score = value as number;
                  const barColor = score >= 0.7 ? "bg-emerald-500" : score >= 0.4 ? "bg-amber-500" : "bg-red-500";
                  return (
                    <div key={key}>
                      <div className="flex justify-between items-center text-xs mb-1">
                        <span className="text-slate-600 font-medium">
                          {SPI_DIMENSION_LABELS[key] || key.replace(/_/g, " ")}
                        </span>
                        <span className="font-mono font-bold text-slate-800">{score.toFixed(3)}</span>
                      </div>
                      <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                        <div
                          className={`${barColor} h-1.5 rounded-full transition-all duration-500`}
                          style={{ width: `${Math.min(score * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Chemistry Properties Table */}
          <div className="col-span-12 lg:col-span-8 bg-white border border-slate-200/80 rounded-xl overflow-hidden shadow-xs">
            <div className="px-4 py-3 border-b border-slate-100 bg-slate-50/50">
              <h3 className="text-xs font-semibold text-slate-800 uppercase tracking-wider font-headline">
                Molecular Properties
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200/60 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    <th className="px-5 py-2.5">Property</th>
                    <th className="px-5 py-2.5">Value</th>
                    <th className="px-5 py-2.5 text-right">Unit</th>
                  </tr>
                </thead>
                <tbody className="text-slate-700 divide-y divide-slate-50 text-xs font-medium">
                  <tr>
                    <td className="px-5 py-3 font-semibold text-slate-500">Molecular Weight</td>
                    <td className="px-5 py-3 font-mono text-slate-900 font-bold">{data.chemistry.molecular_weight.toFixed(2)}</td>
                    <td className="px-5 py-3 text-slate-400 text-right">g/mol</td>
                  </tr>
                  <tr>
                    <td className="px-5 py-3 font-semibold text-slate-500">Heavy Atoms</td>
                    <td className="px-5 py-3 font-mono text-slate-900 font-bold">{data.chemistry.num_heavy_atoms}</td>
                    <td className="px-5 py-3 text-slate-400 text-right font-mono">—</td>
                  </tr>
                  <tr>
                    <td className="px-5 py-3 font-semibold text-slate-500">LogP</td>
                    <td className="px-5 py-3 font-mono text-slate-900 font-bold">{data.chemistry.logp.toFixed(2)}</td>
                    <td className="px-5 py-3 text-slate-400 text-right font-mono">—</td>
                  </tr>
                  <tr>
                    <td className="px-5 py-3 font-semibold text-slate-500">TPSA</td>
                    <td className="px-5 py-3 font-mono text-slate-900 font-bold">{data.chemistry.tpsa.toFixed(1)}</td>
                    <td className="px-5 py-3 text-slate-400 text-right">Å²</td>
                  </tr>
                  <tr>
                    <td className="px-5 py-3 font-semibold text-slate-500">Rings</td>
                    <td className="px-5 py-3 font-mono text-slate-900 font-bold">{data.chemistry.num_rings}</td>
                    <td className="px-5 py-3 text-slate-400 text-right font-mono">—</td>
                  </tr>
                  <tr>
                    <td className="px-5 py-3 font-semibold text-slate-500">Max Ring Size</td>
                    <td className="px-5 py-3 font-mono text-slate-900 font-bold">{data.chemistry.max_ring_size}</td>
                    <td className="px-5 py-3 text-slate-400 text-right font-mono">—</td>
                  </tr>
                  <tr>
                    <td className="px-5 py-3 font-semibold text-slate-500">Stereocenters</td>
                    <td className="px-5 py-3 font-mono text-slate-900 font-bold">{data.chemistry.num_stereocenters}</td>
                    <td className="px-5 py-3 text-slate-400 text-right font-mono">—</td>
                  </tr>
                  <tr>
                    <td className="px-5 py-3 font-semibold text-slate-500">Rotatable Bonds</td>
                    <td className="px-5 py-3 font-mono text-slate-900 font-bold">{data.chemistry.num_rotatable_bonds}</td>
                    <td className="px-5 py-3 text-slate-400 text-right font-mono">—</td>
                  </tr>
                  <tr>
                    <td className="px-5 py-3 font-semibold text-slate-500">Macrocycle</td>
                    <td className="px-5 py-3 font-mono text-slate-900 font-bold">{data.chemistry.is_macrocycle ? "Yes" : "No"}</td>
                    <td className="px-5 py-3 text-slate-400 text-right font-mono">—</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* SPI Report (collapsible) */}
          <div className="col-span-12 lg:col-span-4 bg-white border border-slate-200/80 rounded-xl shadow-xs overflow-hidden">
            <button
              onClick={() => setShowReport(!showReport)}
              className="w-full px-4 py-3 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center text-xs font-semibold text-slate-800 uppercase tracking-wider font-headline hover:bg-slate-50 transition-colors"
            >
              <span>SPI Report</span>
              {showReport ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </button>
            {showReport && (
              <div className="p-4">
                <pre className="text-[10px] font-mono text-slate-600 whitespace-pre-wrap leading-relaxed overflow-x-auto">
                  {data.spi_report}
                </pre>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Empty state */
        <div className="bg-white border border-slate-100 rounded-2xl p-12 flex flex-col items-center justify-center text-center gap-4 py-20 shadow-xs border-dashed">
          <div className="w-16 h-16 rounded-2xl bg-slate-100 flex items-center justify-center">
            <Sparkles size={28} className="text-slate-400" />
          </div>
          <div>
            <h3 className="font-bold text-slate-800 text-base">No Prediction Yet</h3>
            <p className="text-slate-400 text-xs mt-1 max-w-sm mx-auto">
              Enter a SMILES string above or select a preset molecule to run the SynFeasNet inference pipeline.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
