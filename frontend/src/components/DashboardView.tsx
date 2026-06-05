/**
 * DashboardView.tsx — Platform Overview Dashboard
 *
 * Shows quick-select molecules and recent activity. When a user clicks
 * a molecule, it triggers a real prediction via the backend.
 */

import { Database, TrendingUp, BarChart2, ShieldCheck, ArrowRight, FlaskConical } from "lucide-react";

interface DashboardViewProps {
  onSelectSMILES: (smiles: string) => void;
  onNavigate: (tabId: string) => void;
}

interface QuickMolecule {
  id: string;
  name: string;
  smiles: string;
  description: string;
}

export default function DashboardView({ onSelectSMILES, onNavigate }: DashboardViewProps) {
  const quickMolecules: QuickMolecule[] = [
    { id: "SYN-001", name: "Ethanol", smiles: "CCO", description: "Simple alcohol — trivial synthesis" },
    { id: "SYN-002", name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O", description: "Acetylsalicylic acid — ester" },
    { id: "SYN-003", name: "Caffeine", smiles: "Cn1cnc2c1c(=O)n(c(=O)n2C)C", description: "Xanthine alkaloid" },
    { id: "SYN-004", name: "Ibuprofen", smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O", description: "NSAID — propionic acid" },
    { id: "SYN-005", name: "Paracetamol", smiles: "CC(=O)Nc1ccc(O)cc1", description: "Acetaminophen" },
    { id: "SYN-006", name: "Benzoic acid", smiles: "OC(=O)c1ccccc1", description: "Simple aromatic acid" },
  ];

  return (
    <div className="flex flex-col gap-6 w-full animate-fade-in">
      {/* Page Header */}
      <div className="flex justify-between items-end border-b border-slate-100 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight font-headline">Overview</h1>
          <p className="text-sm text-slate-500 mt-1">
            Quick access to molecule analysis. Select a molecule to run real SPI prediction.
          </p>
        </div>
      </div>

      {/* Info cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-white border border-slate-200/80 rounded-xl p-5 flex flex-col justify-between shadow-xs hover:border-blue-300 transition-colors">
          <div className="flex justify-between items-start">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Model Architecture</span>
            <div className="p-1 rounded-md bg-blue-50 text-blue-600">
              <Database size={16} />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-xl font-bold text-slate-800 tracking-tight">4 Branches</span>
            <p className="text-xs text-slate-500 mt-1">ANN + GAT + ChemBERTa + EGNN</p>
          </div>
        </div>

        <div className="bg-white border border-slate-200/80 rounded-xl p-5 flex flex-col justify-between shadow-xs hover:border-amber-300 transition-colors">
          <div className="flex justify-between items-start">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">SPI Dimensions</span>
            <div className="p-1 rounded-md bg-amber-50 text-amber-600">
              <BarChart2 size={16} />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-xl font-bold text-slate-800 tracking-tight">6 Sub-Scores</span>
            <p className="text-xs text-slate-500 mt-1">Multi-dimensional practicality</p>
          </div>
        </div>

        <div className="bg-white border border-slate-200/80 rounded-xl p-5 flex flex-col justify-between shadow-xs hover:border-emerald-300 transition-colors">
          <div className="flex justify-between items-start">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Retrosynthesis</span>
            <div className="p-1 rounded-md bg-emerald-50 text-emerald-600">
              <ShieldCheck size={16} />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-xl font-bold text-slate-800 tracking-tight">Template-Based</span>
            <p className="text-xs text-slate-500 mt-1">Ester, amide, ether cleavage</p>
          </div>
        </div>
      </div>

      {/* Quick Analyze Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-8 bg-white border border-slate-200/80 rounded-xl overflow-hidden shadow-xs">
          <div className="px-5 py-4 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
            <h2 className="text-sm font-semibold text-slate-800">Quick Analyze</h2>
            <button
              onClick={() => onNavigate("molecule")}
              className="text-xs font-semibold text-blue-600 hover:text-blue-700 flex items-center gap-1 transition-colors"
            >
              Custom SMILES <ArrowRight size={14} />
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200/60 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  <th className="px-5 py-3">ID</th>
                  <th className="px-5 py-3">Name</th>
                  <th className="px-5 py-3">SMILES</th>
                  <th className="px-5 py-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm font-medium">
                {quickMolecules.map((mol) => (
                  <tr
                    key={mol.id}
                    className="hover:bg-slate-50/80 transition-colors cursor-pointer group"
                    onClick={() => onSelectSMILES(mol.smiles)}
                  >
                    <td className="px-5 py-3.5 font-mono text-xs text-blue-600 font-semibold group-hover:underline">
                      {mol.id}
                    </td>
                    <td className="px-5 py-3.5">
                      <div>
                        <p className="text-slate-800 font-semibold text-xs">{mol.name}</p>
                        <p className="text-slate-400 text-[10px]">{mol.description}</p>
                      </div>
                    </td>
                    <td className="px-5 py-3.5 font-mono text-xs text-slate-400 max-w-[220px] truncate" title={mol.smiles}>
                      {mol.smiles}
                    </td>
                    <td className="px-5 py-3.5 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectSMILES(mol.smiles);
                        }}
                        className="inline-flex items-center gap-1 px-2.5 py-1 bg-blue-50 text-blue-700 border border-blue-200/50 rounded-md text-[10px] font-bold hover:bg-blue-100 transition-colors"
                      >
                        <FlaskConical size={11} /> Analyze
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Sidebar */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
            <h2 className="text-sm font-semibold text-slate-800 mb-3 font-headline">SPI Score Scale</h2>
            <div className="space-y-2">
              {[
                { label: "Trivial (0.75–1.0)", color: "bg-blue-500", desc: "Standard building blocks" },
                { label: "Practical (0.55–0.75)", color: "bg-emerald-500", desc: "Feasible in med chem lab" },
                { label: "Challenging (0.35–0.55)", color: "bg-amber-500", desc: "Specialist knowledge needed" },
                { label: "Difficult (0.15–0.35)", color: "bg-orange-500", desc: "Significant expertise" },
                { label: "Intractable (0–0.15)", color: "bg-red-500", desc: "Not practical" },
              ].map((item, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className={`w-3 h-3 rounded-full ${item.color} shrink-0`} />
                  <div>
                    <p className="text-xs font-semibold text-slate-700">{item.label}</p>
                    <p className="text-[10px] text-slate-400">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={() => onNavigate("metrics")}
            className="w-full py-2.5 bg-slate-50 hover:bg-slate-100 rounded-lg text-xs font-semibold text-slate-600 border border-slate-200 transition-all text-center"
          >
            View Model Metrics
          </button>
        </div>
      </div>
    </div>
  );
}
