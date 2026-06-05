import { useState, useEffect, useCallback } from "react";
import {
  Layers,
  Search,
  BookOpen,
  BarChart2,
  GitFork,
  Activity,
  Bell,
  Settings,
  Flame,
  Cpu,
} from "lucide-react";

import { PredictionResult, HealthStatus } from "./types";
import DashboardView from "./components/DashboardView";
import MoleculeAnalysisView from "./components/MoleculeAnalysisView";
import RetrosynthesisView from "./components/RetrosynthesisView";
import ModelMetricsView from "./components/ModelMetricsView";
import DocumentationView from "./components/DocumentationView";

const API_BASE = ""; // Vite proxy handles /api → backend

export default function App() {
  const [activeTab, setActiveTab] = useState<string>("dashboard");
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [searchText, setSearchText] = useState<string>("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");

  // Check backend health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/health`);
        if (res.ok) {
          const data: HealthStatus = await res.json();
          setBackendStatus(data.model_loaded ? "online" : "offline");
        } else {
          setBackendStatus("offline");
        }
      } catch {
        setBackendStatus("offline");
      }
    };
    checkHealth();
  }, []);

  // Core prediction handler — calls the real FastAPI backend
  const handleAnalyzeMolecule = useCallback(async (smiles: string) => {
    console.log("[handleAnalyzeMolecule] Predict button clicked / function invoked with SMILES:", smiles);
    if (!smiles) {
      console.log("[handleAnalyzeMolecule] Validation failed: empty SMILES");
      return;
    }
    
    console.log("[handleAnalyzeMolecule] Validation passed, setting loading state...");
    setLoading(true);
    setErrorMsg(null);

    try {
      const targetUrl = `${API_BASE}/api/predict`;
      console.log(`[handleAnalyzeMolecule] Sending API request to: ${targetUrl}`);
      
      const response = await fetch(targetUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles, include_retrosynthesis: true }),
      });

      console.log(`[handleAnalyzeMolecule] Response received, status: ${response.status}`);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        console.error("[handleAnalyzeMolecule] Backend returned error:", errData);
        throw new Error(errData.detail || errData.error || `Server returned ${response.status}`);
      }

      console.log("[handleAnalyzeMolecule] Parsing JSON response...");
      const data: PredictionResult = await response.json();
      
      console.log("[handleAnalyzeMolecule] UI updating with new prediction result:", data);
      setPredictionResult(data);
    } catch (err: any) {
      console.error("[handleAnalyzeMolecule] Exception caught during prediction:", err);
      setErrorMsg(err.message || "Failed to connect to the SynFeasNet backend.");
    } finally {
      console.log("[handleAnalyzeMolecule] Clearing loading state...");
      setLoading(false);
    }
  }, []);

  const handleSearchKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && searchText.trim()) {
      handleAnalyzeMolecule(searchText.trim());
      setActiveTab("molecule");
    }
  };

  const statusColor =
    backendStatus === "online" ? "bg-emerald-500" :
    backendStatus === "offline" ? "bg-red-500" :
    "bg-amber-500";

  const statusText =
    backendStatus === "online" ? "Model Online" :
    backendStatus === "offline" ? "Backend Offline" :
    "Connecting...";

  return (
    <div id="syn_feas_root" className="flex h-screen w-screen bg-slate-50 font-sans overflow-hidden">
      {/* 1. Left Navigation Rail Sidebar */}
      <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col flex-shrink-0 text-slate-300">
        {/* Rail Header */}
        <div className="p-6 border-b border-slate-800 flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-md">
            <Flame className="animate-pulse duration-1000 text-amber-300" size={18} fill="currentColor" />
          </div>
          <div>
            <h1 className="text-white font-bold text-base tracking-tight font-headline">SynFeasNet</h1>
            <p className="text-[10px] text-slate-500 font-extrabold tracking-wider uppercase font-headline">
              SPI Prediction Platform
            </p>
          </div>
        </div>

        {/* Rail Links */}
        <nav className="flex-1 px-4 py-6 space-y-1.5 overflow-y-auto">
          {/* Dashboard */}
          <button
            onClick={() => setActiveTab("dashboard")}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
              activeTab === "dashboard"
                ? "bg-blue-600 text-white shadow-xs"
                : "hover:bg-slate-800 hover:text-white"
            }`}
          >
            <Layers size={16} />
            <span>Dashboard</span>
          </button>

          {/* Molecule Analysis */}
          <button
            onClick={() => setActiveTab("molecule")}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
              activeTab === "molecule"
                ? "bg-blue-600 text-white shadow-xs"
                : "hover:bg-slate-800 hover:text-white"
            }`}
          >
            <Activity size={16} />
            <span>Molecule Analysis</span>
          </button>

          {/* Retrosynthesis */}
          <button
            onClick={() => setActiveTab("retro")}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
              activeTab === "retro"
                ? "bg-blue-600 text-white shadow-xs"
                : "hover:bg-slate-800 hover:text-white"
            }`}
          >
            <GitFork size={16} />
            <span>Retrosynthesis</span>
          </button>

          {/* Model Metrics */}
          <button
            onClick={() => setActiveTab("metrics")}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
              activeTab === "metrics"
                ? "bg-blue-600 text-white shadow-xs"
                : "hover:bg-slate-800 hover:text-white"
            }`}
          >
            <BarChart2 size={16} />
            <span>Model Metrics</span>
          </button>

          {/* Documentation */}
          <button
            onClick={() => setActiveTab("docs")}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
              activeTab === "docs"
                ? "bg-blue-600 text-white shadow-xs"
                : "hover:bg-slate-800 hover:text-white"
            }`}
          >
            <BookOpen size={16} />
            <span>Documentation</span>
          </button>

          <div className="pt-6 border-t border-slate-800 mt-6">
            <button
              onClick={() => {
                setActiveTab("molecule");
                handleAnalyzeMolecule("CC(=O)Oc1ccccc1C(=O)O");
              }}
              className="w-full h-10 px-4 bg-slate-800 hover:bg-slate-700 hover:text-white border border-slate-700/60 rounded-lg text-xs font-semibold text-slate-300 flex items-center justify-center gap-2 transition-all active:scale-[0.98] shadow-sm"
            >
              <Cpu size={14} className="text-blue-400" />
              <span>Run Inference (Aspirin)</span>
            </button>
          </div>
        </nav>

        {/* System status and profile */}
        <div className="p-4 border-t border-slate-800 space-y-4">
          <div className="flex items-center gap-2.5 px-2 py-1 text-xs text-slate-500 font-semibold select-none">
            <span className={`w-2.5 h-2.5 rounded-full ${statusColor} ${backendStatus === "online" ? "animate-pulse" : ""} inline-block`} />
            <span>{statusText}</span>
          </div>

          <div className="flex items-center gap-3 p-2 bg-slate-950/40 border border-slate-800/80 rounded-lg select-none">
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-bold text-sm">
              R
            </div>
            <div className="truncate">
              <p className="text-xs font-bold text-white text-ellipsis overflow-hidden">Researcher</p>
              <p className="text-[10px] text-slate-500 font-semibold truncate leading-none mt-1">SynFeasNet User</p>
            </div>
          </div>
        </div>
      </aside>

      {/* 2. Primary Workspace Panel */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Upper Search & Notification Navbar */}
        <header className="h-16 bg-white border-b border-slate-200 px-6 flex items-center justify-between flex-shrink-0 shadow-3xs z-10">
          {/* Quick search input */}
          <div className="relative w-80 max-w-md">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none">
              <Search size={16} />
            </span>
            <input
              type="text"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              onKeyDown={handleSearchKeyPress}
              placeholder="Enter SMILES string and press Enter..."
              className="w-full h-9 pl-9 pr-4 text-xs bg-slate-50/70 hover:bg-slate-50 border border-slate-200 rounded-lg focus:bg-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-hidden transition-all text-slate-500"
            />
          </div>

          {/* Right widgets */}
          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold tracking-wide select-none border ${
              backendStatus === "online"
                ? "bg-emerald-50 border-emerald-100 text-emerald-700"
                : "bg-red-50 border-red-100 text-red-700"
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${statusColor} ${backendStatus === "online" ? "animate-ping" : ""} inline-block`} />
              <span>{statusText}</span>
            </div>

            <button className="p-1.5 hover:bg-slate-100 rounded-md text-slate-500 transition-colors relative">
              <Bell size={18} />
            </button>

            <button className="p-1.5 hover:bg-slate-100 rounded-md text-slate-500 transition-colors">
              <Settings size={18} />
            </button>
          </div>
        </header>

        {/* 3. Main Route Content view switcher */}
        <main className="flex-1 overflow-y-auto bg-slate-50 p-6">
          <div className="max-w-7xl mx-auto h-full">
            {activeTab === "dashboard" && (
              <DashboardView
                onSelectSMILES={(smiles) => {
                  handleAnalyzeMolecule(smiles);
                  setActiveTab("molecule");
                }}
                onNavigate={(tabId) => {
                  setActiveTab(tabId);
                }}
              />
            )}
            {activeTab === "molecule" && (
              <MoleculeAnalysisView
                predictionResult={predictionResult}
                loading={loading}
                onAnalyze={(smiles) => handleAnalyzeMolecule(smiles)}
                errorMsg={errorMsg}
              />
            )}
            {activeTab === "retro" && (
              <RetrosynthesisView predictionResult={predictionResult} />
            )}
            {activeTab === "metrics" && <ModelMetricsView />}
            {activeTab === "docs" && <DocumentationView />}
          </div>
        </main>
      </div>
    </div>
  );
}
