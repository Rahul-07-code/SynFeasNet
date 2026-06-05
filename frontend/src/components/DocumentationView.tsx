import { Book, HelpCircle, Shield, Code, Cpu } from "lucide-react";

export default function DocumentationView() {
  return (
    <div className="flex flex-col gap-6 w-full max-w-4xl mx-auto animate-fade-in text-slate-800">
      {/* Page Header */}
      <div className="border-b border-slate-100 pb-4">
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight font-headline">Platform Documentation</h1>
        <p className="text-sm text-slate-500 mt-1">Understanding the SynFeasNet active synthetic feasibility scoring algorithms.</p>
      </div>

      <div className="space-y-6">
        {/* Section 1 */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2 mb-3 font-headline">
            <Cpu className="text-blue-500" size={18} /> Model Architecture (v4.2.0-beta)
          </h2>
          <p className="text-slate-600 text-xs leading-relaxed">
            SynFeasNet integrates deep graph neural networks with standard molecular descriptor indices. By processing molecular graph connections (bonds) as edge features and atoms as node vectors, the model extracts high-dimensional embeddings that accurately predict:
          </p>
          <ul className="list-disc pl-5 text-xs text-slate-600 space-y-1 mt-3 font-medium">
            <li><strong>Synthetic Accessibility:</strong> The likelihood of successfully synthesising the target from raw catalog molecules (0.0=Very Hard to 1.0=Immediate).</li>
            <li><strong>Retrosynthesis Step Paths:</strong> Feasible multi-step reactions utilizing standard catalog reagents and optimal catalysts.</li>
            <li><strong>Stereochemical Conformers:</strong> Calculated 3D coordinates based on energy minimization predictions.</li>
          </ul>
        </div>

        {/* Section 2 */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2 mb-3 font-headline">
            <Book className="text-blue-500" size={18} /> SMILES Format Standard
          </h2>
          <p className="text-slate-600 text-xs leading-relaxed">
            The platform supports standard Simplified Molecular Input Line Entry System (SMILES) syntax. 
            For best performance, enter finalized canonical SMILES inputs. Below are common examples:
          </p>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 font-mono text-xs text-slate-700 mt-3 space-y-1.5 shadow-inner">
            <p><strong>Paracetamol:</strong> CC(=O)NC1=CC=C(O)C=C1</p>
            <p><strong>Aspirin:</strong> CC(=O)OC1=CC=CC=C1C(=O)O</p>
            <p><strong>Caffeine:</strong> CN1C=NC2=C1C(=O)N(C(=O)N2C)C</p>
          </div>
        </div>

        {/* Section 3 FAQ */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-2xs">
          <h3 className="text-sm font-bold text-slate-900 mb-3 font-headline flex items-center gap-1.5">
            <Shield className="text-blue-500" size={16} /> Data Security & Compliance
          </h3>
          <p className="text-slate-600 text-xs leading-relaxed">
            All submitted SMILES structural inputs are analyzed in real-time server-side using secure proxy layers. No proprietary structural formulations or custom candidate molecules are logged or persisted in cold storage without explicit administrative permission.
          </p>
        </div>
      </div>
    </div>
  );
}
