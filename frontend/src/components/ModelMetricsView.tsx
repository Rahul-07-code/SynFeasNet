/**
 * ModelMetricsView.tsx — Model Metrics from Backend
 *
 * Fetches real metrics from GET /api/metrics and displays:
 * - Parameter counts per branch
 * - Checkpoint metadata (epoch, val_spi_mae, val_spearman)
 * - SPI dimension weights
 * - SPI class labels
 */

import { useState, useEffect } from "react";
import { FileText, Loader2, RefreshCw } from "lucide-react";
import { ModelMetrics } from "../types";

export default function ModelMetricsView() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/metrics");
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data: ModelMetrics = await res.json();
      setMetrics(data);
    } catch (err: any) {
      setError(err.message || "Failed to fetch metrics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const formatNumber = (n: number) => n.toLocaleString();

  return (
    <div className="flex flex-col gap-6 w-full animate-fade-in text-slate-800">
      {/* Page Header */}
      <div className="flex justify-between items-end border-b border-slate-100 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight font-headline">Model Metrics</h1>
          <p className="text-sm text-slate-500 mt-1">
            Real-time model configuration and parameter counts from the SynFeasNet backend.
          </p>
        </div>
        <button
          onClick={fetchMetrics}
          disabled={loading}
          className="bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold h-9 px-3 border border-slate-200 rounded-lg flex items-center gap-1.5 transition-colors disabled:opacity-50"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} /> Refresh
        </button>
      </div>

      {loading && !metrics ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 size={24} className="animate-spin text-blue-500" />
          <span className="ml-2 text-sm text-slate-500">Loading metrics from backend...</span>
        </div>
      ) : error ? (
        <div className="bg-red-50 border border-red-200 rounded-xl p-5 text-red-700 text-sm">
          <p className="font-bold">Failed to load metrics</p>
          <p className="text-xs mt-1">{error}</p>
          <p className="text-xs mt-2 text-red-500">Make sure the FastAPI backend is running on port 8000.</p>
        </div>
      ) : metrics ? (
        <>
          {/* Checkpoint Info */}
          {metrics.checkpoint_info && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
              {metrics.checkpoint_info.epoch !== null && metrics.checkpoint_info.epoch !== undefined && (
                <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-2xs hover:border-blue-400 transition-colors">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Training Epoch</span>
                  <div className="text-2xl font-black font-headline text-slate-900 mt-2">
                    {String(metrics.checkpoint_info.epoch)}
                  </div>
                </div>
              )}
              {metrics.checkpoint_info.val_spi_mae !== null && metrics.checkpoint_info.val_spi_mae !== undefined && (
                <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-2xs hover:border-emerald-400 transition-colors">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Val SPI MAE</span>
                  <div className="text-2xl font-black font-headline text-emerald-600 mt-2">
                    {Number(metrics.checkpoint_info.val_spi_mae).toFixed(4)}
                  </div>
                </div>
              )}
              {metrics.checkpoint_info.val_spearman !== null && metrics.checkpoint_info.val_spearman !== undefined && (
                <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-2xs hover:border-blue-400 transition-colors">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Val Spearman ρ</span>
                  <div className="text-2xl font-black font-headline text-blue-600 mt-2">
                    {Number(metrics.checkpoint_info.val_spearman).toFixed(4)}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Parameter Counts */}
          {metrics.parameter_counts && (
            <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
              <div className="px-4 py-3 border-b border-slate-150 bg-slate-50/50">
                <h3 className="text-xs font-bold text-slate-750 uppercase tracking-widest font-headline">
                  Parameter Counts by Branch
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-150 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      <th className="py-3 px-5">Branch / Component</th>
                      <th className="py-3 px-5 text-right">Parameters</th>
                      <th className="py-3 px-5 text-right">% of Total</th>
                    </tr>
                  </thead>
                  <tbody className="text-xs font-medium divide-y divide-slate-100 text-slate-700">
                    {Object.entries(metrics.parameter_counts)
                      .filter(([key]) => key !== "total" && key !== "trainable")
                      .map(([key, count]) => {
                        const total = metrics.parameter_counts?.total || 1;
                        const pct = ((count / total) * 100).toFixed(1);
                        return (
                          <tr key={key} className="hover:bg-slate-50/80 transition-colors">
                            <td className="py-3.5 px-5 font-semibold text-slate-800 capitalize flex items-center gap-2">
                              <span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block" />
                              {key}
                            </td>
                            <td className="py-3.5 px-5 text-right font-mono">{formatNumber(count)}</td>
                            <td className="py-3.5 px-5 text-right font-mono text-slate-400">{pct}%</td>
                          </tr>
                        );
                      })}
                    {/* Total row */}
                    <tr className="bg-slate-50 font-bold">
                      <td className="py-3.5 px-5 text-slate-900">Total</td>
                      <td className="py-3.5 px-5 text-right font-mono text-slate-900">
                        {formatNumber(metrics.parameter_counts.total)}
                      </td>
                      <td className="py-3.5 px-5 text-right font-mono text-slate-400">100%</td>
                    </tr>
                    <tr className="bg-blue-50/50">
                      <td className="py-3.5 px-5 text-blue-700 font-semibold">Trainable</td>
                      <td className="py-3.5 px-5 text-right font-mono text-blue-700">
                        {formatNumber(metrics.parameter_counts.trainable)}
                      </td>
                      <td className="py-3.5 px-5 text-right font-mono text-blue-400">
                        {((metrics.parameter_counts.trainable / metrics.parameter_counts.total) * 100).toFixed(1)}%
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* SPI Weights */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
              <h3 className="text-xs font-bold text-slate-750 uppercase tracking-widest font-headline mb-4">
                SPI Dimension Weights
              </h3>
              <div className="space-y-3">
                {Object.entries(metrics.spi_weights).map(([dim, weight]) => (
                  <div key={dim}>
                    <div className="flex justify-between items-center text-xs mb-1">
                      <span className="text-slate-600 font-medium capitalize">
                        {dim.replace(/_/g, " ")}
                      </span>
                      <span className="font-mono font-bold text-slate-800">{(weight * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-blue-500 h-2 rounded-full"
                        style={{ width: `${weight * 100 * 5}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
              <h3 className="text-xs font-bold text-slate-750 uppercase tracking-widest font-headline mb-4">
                SPI Class Labels
              </h3>
              <div className="space-y-2.5">
                {metrics.spi_class_labels.map((label, idx) => {
                  const colors = [
                    "bg-red-100 text-red-700 border-red-200",
                    "bg-orange-100 text-orange-700 border-orange-200",
                    "bg-amber-100 text-amber-700 border-amber-200",
                    "bg-emerald-100 text-emerald-700 border-emerald-200",
                    "bg-blue-100 text-blue-700 border-blue-200",
                  ];
                  return (
                    <div
                      key={label}
                      className={`flex items-center justify-between px-3 py-2 rounded-lg border text-xs font-semibold ${colors[idx]}`}
                    >
                      <span className="capitalize">{label}</span>
                      <span className="font-mono">Class {idx}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
