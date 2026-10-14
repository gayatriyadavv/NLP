import React from 'react';
import { Network, FileText, Search, Sparkles, Cpu, Layers } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  backendStatus: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, backendStatus }) => {
  const navItems = [
    { id: 'landing', label: 'Overview', icon: Sparkles },
    { id: 'workspace', label: 'Upload Workspace', icon: FileText },
    { id: 'analysis', label: 'Analysis Workspace', icon: Layers },
    { id: 'graph', label: 'Entity Grounding Graph', icon: Network },
    { id: 'search', label: 'Cross-Modal Search', icon: Search }
  ];

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800 px-6 py-3.5 flex items-center justify-between">
      {/* Brand Title */}
      <div 
        onClick={() => setActiveTab('landing')}
        className="flex items-center gap-3 cursor-pointer group"
      >
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500 to-indigo-600 p-0.5 flex items-center justify-center shadow-lg shadow-cyan-500/20 group-hover:shadow-cyan-500/40 transition-all duration-300">
          <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
            <Cpu className="w-5 h-5 text-cyan-400 group-hover:scale-110 transition-transform" />
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-semibold tracking-tight text-slate-100 text-sm md:text-base">
              Multi-Modal Scientific Visualization Understanding
            </h1>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-500/30 text-cyan-300">
              PROTOTYPE
            </span>
          </div>
          <p className="text-[11px] text-slate-400 font-mono">Hierarchical Context Grounding & Cross-Modal Retrieval</p>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="hidden lg:flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-gradient-to-r from-cyan-500/20 to-indigo-500/20 text-cyan-300 border border-cyan-500/30 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Backend Status Indicator */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono">
          <span className={`w-2 h-2 rounded-full ${backendStatus ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
          <span className="text-slate-300 text-[11px]">
            {backendStatus ? 'Backend Online' : 'Backend Offline (demo data)'}
          </span>
        </div>
      </div>
    </header>
  );
};
