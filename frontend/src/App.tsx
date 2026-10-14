import { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { LandingPage } from './components/LandingPage';
import { UploadWorkspace } from './components/UploadWorkspace';
import { MainDashboard } from './components/MainDashboard';
import { GroundingGraphView } from './components/GroundingGraphView';
import { CrossModalSearch } from './components/CrossModalSearch';
import type { PipelineResult } from './types';
import { DEFAULT_PIPELINE_RESULT } from './data/mockData';
import { checkBackendHealth } from './services/api';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('landing');
  const [pipelineResult, setPipelineResult] = useState<PipelineResult>(DEFAULT_PIPELINE_RESULT);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);

  useEffect(() => {
    checkBackendHealth().then(setBackendOnline);
  }, []);

  const handlePipelineSuccess = (result: PipelineResult) => {
    setPipelineResult(result);
    setActiveTab('analysis');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500 selection:text-slate-950">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendStatus={backendOnline}
      />

      {pipelineResult.is_mock_data && activeTab !== 'landing' && (
        <div className="max-w-7xl mx-auto px-6 pt-4">
          <div className="rounded-xl border border-amber-500/40 bg-amber-950/40 px-4 py-2 text-sm text-amber-200">
            Showing built-in demo data, not a backend result. Numbers on this page were not computed.
          </div>
        </div>
      )}

      <main className="pb-12">
        {activeTab === 'landing' && (
          <LandingPage
            onStartAnalysis={() => setActiveTab('workspace')}
            onExploreDemo={() => setActiveTab('analysis')}
          />
        )}

        {activeTab === 'workspace' && (
          <UploadWorkspace onPipelineSuccess={handlePipelineSuccess} />
        )}

        {activeTab === 'analysis' && (
          <MainDashboard pipelineResult={pipelineResult} />
        )}

        {activeTab === 'graph' && (
          <div className="max-w-7xl mx-auto px-6 py-8 h-[calc(100vh-100px)]">
            <GroundingGraphView graphSummary={pipelineResult.grounding_graph_summary} />
          </div>
        )}

        {activeTab === 'search' && <CrossModalSearch />}
      </main>
    </div>
  );
}

export default App;
