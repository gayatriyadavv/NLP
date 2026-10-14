import React, { useState } from 'react';
import { Layers, FileText, Cpu, Network, Sparkles, ShieldCheck, BookOpen } from 'lucide-react';
import type { PipelineResult, DocumentStructure } from '../types';
import { SAMPLE_DOCUMENT } from '../data/mockData';
import { FigureViewer } from './FigureViewer';
import { HierarchicalContextView } from './HierarchicalContextView';
import { EntityExplorer } from './EntityExplorer';
import { GroundingGraphView } from './GroundingGraphView';
import { CaptionPanel } from './CaptionPanel';
import { VerificationPanel } from './VerificationPanel';

interface MainDashboardProps {
  pipelineResult: PipelineResult;
}

export const MainDashboard: React.FC<MainDashboardProps> = ({ pipelineResult }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'context' | 'entities' | 'graph' | 'caption' | 'verification'>('overview');
  const doc: DocumentStructure = pipelineResult.document_structure ?? SAMPLE_DOCUMENT;
  const isLive = !pipelineResult.is_mock_data && !!pipelineResult.figure_path;

  return (
    <div className="max-w-[1600px] mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[calc(100vh-80px)]">
      {/* LEFT PANEL: Paper Navigation (3 cols) */}
      <div className="lg:col-span-3 glass-panel rounded-xl p-4 border border-slate-800 flex flex-col space-y-4">
        <div className="pb-3 border-b border-slate-800">
          <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider block">Document Navigation</span>
          <h3 className="text-xs font-semibold text-slate-100 font-mono mt-1 line-clamp-2">
            {doc.title}
          </h3>
          <span className="text-[10px] text-slate-400 font-mono">ID: {doc.paper_id}{doc.num_pages ? ` • ${doc.num_pages} Pages` : ''}</span>
        </div>

        {/* Section Tree */}
        <div className="flex-1 overflow-y-auto space-y-3 font-mono text-xs pr-1">
          {doc.sections.map((sec) => (
            <div key={sec.section_id} className="space-y-1">
              <div className="flex items-center gap-1.5 text-cyan-400 font-semibold py-1 px-2 rounded bg-slate-900/80 border border-slate-800/80 text-[11px]">
                <BookOpen className="w-3.5 h-3.5 text-indigo-400" />
                <span className="truncate">{sec.title}</span>
              </div>
              <div className="pl-3 space-y-1">
                {sec.paragraphs.map((p) => (
                  <div key={p.paragraph_id} className="p-2 rounded bg-slate-950/60 border border-slate-800/50 text-[11px] text-slate-300">
                    <span className="text-cyan-400 font-semibold block text-[10px]">[{p.paragraph_id}] Page {p.page}</span>
                    <p className="line-clamp-2 text-slate-400 font-sans text-[11px] mt-0.5">{p.text}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* CENTER PANEL: Large Figure Viewer (5 cols) */}
      <div className="lg:col-span-5 flex flex-col">
        <FigureViewer
          elements={pipelineResult.figure_elements.elements}
          threedAnalysis={pipelineResult['3d_branch_analysis']}
          imageUrl={isLive ? `/api/figure-image?path=${encodeURIComponent(pipelineResult.figure_path!)}` : undefined}
          dimensions={pipelineResult.figure_elements.dimensions}
          graphSummary={isLive ? pipelineResult.grounding_graph_summary : undefined}
        />
      </div>

      {/* RIGHT PANEL: AI Analysis Tabs (4 cols) */}
      <div className="lg:col-span-4 glass-panel rounded-xl p-4 border border-slate-800 flex flex-col">
        {/* Analysis Tab Switcher */}
        <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800 mb-4 overflow-x-auto">
          {[
            { id: 'overview', label: 'Overview', icon: Sparkles },
            { id: 'context', label: 'Context', icon: Layers },
            { id: 'entities', label: 'Entities', icon: Cpu },
            { id: 'graph', label: 'Graph', icon: Network },
            { id: 'caption', label: 'Caption', icon: FileText },
            { id: 'verification', label: 'Verify', icon: ShieldCheck }
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11px] font-mono transition-all flex-shrink-0 cursor-pointer ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/20 to-indigo-500/20 text-cyan-300 border border-cyan-500/30 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Tab Content Rendering */}
        <div className="flex-1 overflow-y-auto pr-1">
          {activeTab === 'overview' && (
            <div className="space-y-4 font-mono text-xs">
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                <span className="text-[10px] text-cyan-400 uppercase tracking-wider block">Query Execution</span>
                <p className="text-slate-200 font-semibold font-sans">"{pipelineResult.query}"</p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">SCQ Score</span>
                  <span className="text-base font-bold text-cyan-400">
                    {(pipelineResult.scq_evaluation.scq_composite_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">Fact Correctness</span>
                  <span className="text-base font-bold text-emerald-400">
                    {(pipelineResult.verification_results.technical_correctness_ratio * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                <span className="text-[10px] text-cyan-400 uppercase tracking-wider block">Generated Grounded Caption</span>
                <p className="text-slate-300 font-sans italic text-xs leading-relaxed">
                  "{pipelineResult.generated_caption}"
                </p>
              </div>
            </div>
          )}

          {activeTab === 'context' && <HierarchicalContextView retrievedContext={pipelineResult.retrieved_context} />}
          {activeTab === 'entities' && <EntityExplorer entities={pipelineResult.extracted_entities} />}
          {activeTab === 'graph' && <GroundingGraphView graphSummary={pipelineResult.grounding_graph_summary} />}
          {activeTab === 'caption' && <CaptionPanel pipelineResult={pipelineResult} />}
          {activeTab === 'verification' && <VerificationPanel pipelineResult={pipelineResult} />}
        </div>
      </div>
    </div>
  );
};
