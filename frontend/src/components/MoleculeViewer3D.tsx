import { useEffect, useRef, useState } from "react";
import * as $3Dmol from "3dmol";
import { Loader2, Maximize, RotateCcw, AlertTriangle, Info } from "lucide-react";

interface MoleculeViewer3DProps {
  smiles: string;
}

interface MolInfo {
  formula: string;
  weight: string;
  atoms: number;
  bonds: number;
}

const ATOM_COLORS: Record<string, string> = {
  C: "gray",
  H: "white",
  O: "red",
  N: "blue",
  S: "yellow",
  Cl: "green",
  F: "lightgreen",
  P: "orange",
  Br: "darkred",
  I: "purple"
};

const ATOMIC_NUMBERS: Record<string, number> = {
  H: 1, C: 6, N: 7, O: 8, F: 9, P: 15, S: 16, Cl: 17, Br: 35, I: 53
};

const ELEMENT_NAMES: Record<string, string> = {
  H: "Hydrogen", C: "Carbon", N: "Nitrogen", O: "Oxygen", F: "Fluorine",
  P: "Phosphorus", S: "Sulfur", Cl: "Chlorine", Br: "Bromine", I: "Iodine"
};

export default function MoleculeViewer3D({ smiles }: MoleculeViewer3DProps) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const viewerInstanceRef = useRef<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  const [molInfo, setMolInfo] = useState<MolInfo | null>(null);
  const [hoveredAtom, setHoveredAtom] = useState<any>(null);

  useEffect(() => {
    if (!smiles || !viewerRef.current) return;

    let isMounted = true;
    setLoading(true);
    setError(null);
    setMolInfo(null);

    // Initialize viewer if not already initialized
    if (!viewerInstanceRef.current) {
      viewerInstanceRef.current = $3Dmol.createViewer(viewerRef.current, {
        backgroundColor: "white",
      });
    }

    const viewer = viewerInstanceRef.current;

    const fetchAndRender3D = async () => {
      try {
        const pubchemUrl = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(smiles)}/SDF?record_type=3d`;
        const res = await fetch(pubchemUrl);

        if (!res.ok) {
          throw new Error(`Failed to generate 3D structure. The molecule might be too complex or not available in the public 3D database.`);
        }

        const sdfContent = await res.text();

        if (!isMounted) return;

        // Parse SDF for basic info
        const lines = sdfContent.split("\n");
        let aCount = 0;
        let bCount = 0;
        const elemCounts: Record<string, number> = {};
        
        if (lines.length > 3) {
          const countsLine = lines[3];
          aCount = parseInt(countsLine.substring(0, 3).trim()) || 0;
          bCount = parseInt(countsLine.substring(3, 6).trim()) || 0;
          
          for (let i = 4; i < 4 + aCount; i++) {
            const elem = lines[i].substring(31, 34).trim();
            if (elem) {
              elemCounts[elem] = (elemCounts[elem] || 0) + 1;
            }
          }
        }

        // Build formula (Hill system approximation)
        let formula = "";
        if (elemCounts["C"]) formula += `C${elemCounts["C"] > 1 ? elemCounts["C"] : ""}`;
        if (elemCounts["H"]) formula += `H${elemCounts["H"] > 1 ? elemCounts["H"] : ""}`;
        Object.keys(elemCounts).sort().forEach(e => {
          if (e !== "C" && e !== "H") {
            formula += `${e}${elemCounts[e] > 1 ? elemCounts[e] : ""}`;
          }
        });

        // Rough MW calculation
        let weight = 0;
        const ATOM_WEIGHTS: Record<string, number> = { C: 12.011, H: 1.008, O: 15.999, N: 14.007, S: 32.06, Cl: 35.45, F: 18.998, P: 30.974, Br: 79.904, I: 126.90 };
        Object.entries(elemCounts).forEach(([e, count]) => {
          if (ATOM_WEIGHTS[e]) weight += ATOM_WEIGHTS[e] * count;
        });

        setMolInfo({
          formula: formula || "Unknown",
          weight: weight > 0 ? weight.toFixed(2) : "Unknown",
          atoms: aCount,
          bonds: bCount
        });

        viewer.clear();
        viewer.addModel(sdfContent, "sdf");
        
        // Professional styling
        viewer.setStyle({}, { stick: { radius: 0.15 }, sphere: { radius: 0.4 } });
        
        // Hover interactions
        viewer.setHoverable({}, true, 
          (atom: any) => {
            if (isMounted) setHoveredAtom(atom);
          },
          () => {
            if (isMounted) setHoveredAtom(null);
          }
        );
        
        viewer.zoomTo();
        viewer.render();
        setLoading(false);
      } catch (err: any) {
        if (!isMounted) return;
        console.error("3D rendering error:", err);
        setError(err.message || "Failed to generate 3D structure");
        setLoading(false);
      }
    };

    fetchAndRender3D();

    return () => {
      isMounted = false;
    };
  }, [smiles]);

  const handleReset = () => {
    if (viewerInstanceRef.current) {
      viewerInstanceRef.current.zoomTo();
    }
  };

  const handleFullscreen = () => {
    if (!viewerRef.current) return;
    if (!document.fullscreenElement) {
      viewerRef.current.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  return (
    <div className="relative w-full h-[600px] bg-white border border-slate-200 rounded-xl overflow-hidden flex flex-col shadow-sm">
      {/* Viewer Canvas */}
      <div 
        ref={viewerRef} 
        className="flex-1 w-full relative"
        style={{ backgroundColor: "#f8fafc" }}
      >
        {/* Loading Overlay */}
        {loading && (
          <div className="absolute inset-0 bg-white/80 backdrop-blur-sm z-10 flex flex-col items-center justify-center">
            <Loader2 className="animate-spin text-blue-600 mb-2" size={32} />
            <p className="text-sm font-semibold text-slate-600">Generating 3D Conformer...</p>
            <p className="text-xs text-slate-400 mt-1 font-mono">{smiles}</p>
          </div>
        )}

        {/* Error Overlay */}
        {error && !loading && (
          <div className="absolute inset-0 bg-white z-10 flex flex-col items-center justify-center p-6 text-center">
            <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center mb-3">
              <AlertTriangle className="text-red-600" size={24} />
            </div>
            <p className="text-sm font-semibold text-slate-800">Failed to render 3D structure</p>
            <p className="text-xs text-slate-500 mt-1 max-w-sm">{error}</p>
          </div>
        )}

        {/* Controls */}
        <div className="absolute top-3 right-3 z-20 flex flex-col gap-2">
          <button
            onClick={handleReset}
            className="p-2 bg-white/90 hover:bg-blue-50 border border-slate-200 rounded-lg text-slate-600 hover:text-blue-600 transition-colors shadow-sm"
            title="Reset View"
          >
            <RotateCcw size={16} />
          </button>
          <button
            onClick={handleFullscreen}
            className="p-2 bg-white/90 hover:bg-blue-50 border border-slate-200 rounded-lg text-slate-600 hover:text-blue-600 transition-colors shadow-sm"
            title="Fullscreen"
          >
            <Maximize size={16} />
          </button>
        </div>

        {/* Atom Legend */}
        <div className="absolute top-3 left-3 z-10 bg-white/90 backdrop-blur border border-slate-200 rounded-lg p-2.5 shadow-sm text-[10px] font-semibold text-slate-600 select-none">
          <div className="flex items-center gap-1.5 mb-1.5 pb-1.5 border-b border-slate-100">
            <Info size={12} className="text-slate-400" />
            <span className="uppercase tracking-wider">Atom Legend</span>
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-gray-500 shadow-inner"></span> Carbon (C)</div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-white border border-gray-300 shadow-inner"></span> Hydrogen (H)</div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-inner"></span> Oxygen (O)</div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-blue-500 shadow-inner"></span> Nitrogen (N)</div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-yellow-400 shadow-inner"></span> Sulfur (S)</div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-green-500 shadow-inner"></span> Chlorine (Cl)</div>
          </div>
        </div>

        {/* Hover Tooltip */}
        {hoveredAtom && (
          <div 
            className="absolute z-30 bg-slate-900 text-white text-xs rounded-lg p-2.5 shadow-lg pointer-events-none transform -translate-x-1/2 -translate-y-full"
            style={{ 
              left: '50%', 
              top: '40px',
              minWidth: '140px'
            }}
          >
            <div className="font-bold mb-1 border-b border-slate-700 pb-1 flex justify-between">
              <span>{ELEMENT_NAMES[hoveredAtom.elem] || hoveredAtom.elem}</span>
              <span className="text-blue-400">{hoveredAtom.elem}</span>
            </div>
            <div className="grid grid-cols-2 gap-x-2 text-[10px] text-slate-300">
              <span>Atomic Number:</span>
              <span className="text-right font-mono text-white">{ATOMIC_NUMBERS[hoveredAtom.elem] || "?"}</span>
              <span>Atom Index:</span>
              <span className="text-right font-mono text-white">{hoveredAtom.serial}</span>
            </div>
          </div>
        )}
      </div>

      {/* Molecular Information Panel */}
      <div className="bg-slate-50 border-t border-slate-200 p-4">
        <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider font-headline mb-3 flex items-center gap-2">
          <Info size={14} className="text-blue-600" /> Molecular Context
        </h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white border border-slate-200 rounded-lg p-2.5 shadow-sm">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-0.5">SMILES</p>
            <p className="text-xs font-mono text-blue-600 truncate" title={smiles}>{smiles}</p>
          </div>
          
          <div className="bg-white border border-slate-200 rounded-lg p-2.5 shadow-sm">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-0.5">Formula</p>
            <p className="text-xs font-semibold text-slate-800">{molInfo?.formula || "—"}</p>
          </div>
          
          <div className="bg-white border border-slate-200 rounded-lg p-2.5 shadow-sm">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-0.5">Molecular Weight</p>
            <p className="text-xs font-semibold text-slate-800">{molInfo?.weight || "—"} <span className="text-slate-400 font-normal">g/mol</span></p>
          </div>
          
          <div className="bg-white border border-slate-200 rounded-lg p-2.5 shadow-sm flex justify-between items-center">
            <div>
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-0.5">Atoms</p>
              <p className="text-xs font-semibold text-slate-800">{molInfo?.atoms || "—"}</p>
            </div>
            <div className="w-px h-6 bg-slate-200"></div>
            <div>
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-0.5">Bonds</p>
              <p className="text-xs font-semibold text-slate-800">{molInfo?.bonds || "—"}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
