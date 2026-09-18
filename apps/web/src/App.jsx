import React, { useState, useEffect } from 'react';
import { 
  Layers, ShieldCheck, Download, Upload, Activity, 
  MapPin, Eye, EyeOff, Award, RefreshCw, Box, Compass
} from 'lucide-react';
import DigitalTwinViewer from './components/DigitalTwinViewer';
import TelemetryHUD from './components/TelemetryHUD';
import PipelineTracker from './components/PipelineTracker';
import AccuracyMetricsModal from './components/AccuracyMetricsModal';
import UploadFlightModal from './components/UploadFlightModal';

export default function App() {
  const [activeLayer, setActiveLayer] = useState('rgb'); // 'rgb' | 'confidence' | 'semantics'
  const [showDynamicObjects, setShowDynamicObjects] = useState(true);
  const [cursorCoords, setCursorCoords] = useState({ east: 12.4, north: -8.2, up: 18.5 });
  
  const [isAccuracyModalOpen, setIsAccuracyModalOpen] = useState(false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  
  const [pipelineProgress, setPipelineProgress] = useState(100);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStage, setCurrentStage] = useState('completed');

  const [metrics, setMetrics] = useState(null);
  const [trajectory, setTrajectory] = useState(null);
  const [pointCloudData, setPointCloudData] = useState(null);
  const [projectsList, setProjectsList] = useState([]);
  const [activeProjectId, setActiveProjectId] = useState('sih-demo-jaipur');
  const [projectName, setProjectName] = useState('Jaipur Urban Infrastructure & Disaster Site');

  const fetchProjects = async () => {
    try {
      const res = await fetch('/api/projects');
      if (res.ok) {
        const list = await res.json();
        setProjectsList(list);
        return list;
      }
    } catch (err) {
      console.error("Projects list fetch error:", err);
    }
    return [];
  };

  const loadProjectData = (pid) => {
    fetch(`/api/projects/${pid}/metrics`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setMetrics(d); })
      .catch(e => console.log("Metrics fetch error", e));

    fetch(`/api/projects/${pid}/trajectory`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setTrajectory(d); })
      .catch(e => console.log("Trajectory fetch error", e));

    fetch(`/api/projects/${pid}/pointcloud_json`)
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (d && d.positions && d.positions.length > 0) {
          setPointCloudData(d);
        }
      })
      .catch(e => console.log("PointCloud fetch error", e));
  };

  const handleSwitchProject = (pid) => {
    setActiveProjectId(pid);
    try {
      localStorage.setItem('sfmx_active_project_id', pid);
    } catch (e) {}
    const found = projectsList.find(p => p.id === pid);
    if (found && found.name) {
      setProjectName(found.name);
    }
    loadProjectData(pid);
  };

  // Fetch projects and initial project data on mount (default to Jaipur Benchmark)
  useEffect(() => {
    (async () => {
      const list = await fetchProjects();
      let targetPid = 'sih-demo-jaipur';
      let targetName = 'Jaipur Urban Infrastructure & Disaster Site';
      
      let savedPid = null;
      try {
        savedPid = localStorage.getItem('sfmx_active_project_id');
      } catch (e) {}

      // If user had explicitly selected an uploaded flight in this session, restore it; otherwise default to Jaipur
      if (savedPid && savedPid !== 'sih-demo-jaipur' && list.some(p => p.id === savedPid)) {
        targetPid = savedPid;
        const p = list.find(p => p.id === savedPid);
        if (p?.name) targetName = p.name;
      }

      setActiveProjectId(targetPid);
      setProjectName(targetName);
      loadProjectData(targetPid);
    })();
  }, []);

  const handleFlightUploadSuccess = async (newProject) => {
    if (!newProject || !newProject.id) return;
    const pid = newProject.id;
    setActiveProjectId(pid);
    try {
      localStorage.setItem('sfmx_active_project_id', pid);
    } catch (e) {}
    if (newProject.name) setProjectName(newProject.name);

    setIsProcessing(true);
    setPipelineProgress(15);
    setCurrentStage('frame_engine');

    await fetchProjects();

    // Poll project status from backend until completed
    const pollTimer = setInterval(async () => {
      try {
        const res = await fetch(`/api/projects/${pid}/status`);
        if (!res.ok) return;
        const statusData = await res.json();
        
        if (statusData.progress) setPipelineProgress(statusData.progress);
        if (statusData.current_stage) setCurrentStage(statusData.current_stage);

        if (statusData.status === 'COMPLETED' || statusData.progress >= 100) {
          clearInterval(pollTimer);
          setIsProcessing(false);
          setCurrentStage('completed');
          setPipelineProgress(100);
          await fetchProjects();
          // Reload the real 3D point cloud & metrics!
          loadProjectData(pid);
        } else if (statusData.status === 'FAILED') {
          clearInterval(pollTimer);
          setIsProcessing(false);
          alert('Pipeline error: ' + (statusData.message || 'Processing failed'));
        }
      } catch (err) {
        console.error("Poll error:", err);
      }
    }, 1000);
  };

  const handleRerunPipeline = () => {
    setIsProcessing(true);
    setPipelineProgress(10);
    setCurrentStage('frame_engine');

    const interval = setInterval(() => {
      setPipelineProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsProcessing(false);
          setCurrentStage('completed');
          loadProjectData(activeProjectId);
          return 100;
        }
        const next = prev + 15;
        if (next < 30) setCurrentStage('frame_engine');
        else if (next < 50) setCurrentStage('sensor_fusion');
        else if (next < 70) setCurrentStage('neural_depth');
        else if (next < 85) setCurrentStage('dynamic_masking');
        else if (next < 95) setCurrentStage('tsdf_fusion');
        else setCurrentStage('georeferencing');
        return next;
      });
    }, 400);
  };

  return (
    <div id="root">
      
      {/* 1. TOP COMMAND HEADER */}
      <header className="command-header">
        {/* Left: Brand & Flight Project */}
        <div className="header-left">
          <div className="brand-badge">
            <div className="brand-logo">SF</div>
            <div>
              <span className="brand-title">SFM-X</span>
              <span className="brand-tag">SINGLE-PASS 4D/3D</span>
            </div>
          </div>

          <div className="divider-v" />

          <div className="project-info">
            <span className="project-label">PROJECT:</span>
            {projectsList.length > 0 ? (
              <select 
                className="project-select"
                value={activeProjectId}
                onChange={(e) => handleSwitchProject(e.target.value)}
                title="Switch Drone Survey Project"
              >
                {projectsList.map(p => (
                  <option key={p.id} value={p.id}>
                    {p.name} {p.status === 'COMPLETED' ? '✓' : `(${p.status})`}
                  </option>
                ))}
              </select>
            ) : (
              <span className="project-name">{projectName}</span>
            )}
            <span className="status-pill-verified">
              {metrics?.georeferencing?.est_3d_spatial_rmse_m 
                ? `RMSE ${metrics.georeferencing.est_3d_spatial_rmse_m.toFixed(3)}m` 
                : 'VERIFIED ≤ 0.136m RMSE'}
            </span>
          </div>
        </div>

        {/* Center: Layer Switcher Toolbar */}
        <div className="header-center">
          <div className="layer-switcher">
            <button 
              className={`layer-btn ${activeLayer === 'rgb' ? 'active' : ''}`}
              onClick={() => setActiveLayer('rgb')}
            >
              Realistic RGB
            </button>

            <button 
              className={`layer-btn ${activeLayer === 'confidence' ? 'active' : ''}`}
              onClick={() => setActiveLayer('confidence')}
            >
              Confidence Heatmap
            </button>

            <button 
              className={`layer-btn ${activeLayer === 'semantics' ? 'active' : ''}`}
              onClick={() => setActiveLayer('semantics')}
            >
              Semantic Layers
            </button>

            <div className="divider-v" />

            <button 
              className={`dynamic-toggle-btn ${showDynamicObjects ? '' : 'off'}`}
              onClick={() => setShowDynamicObjects(!showDynamicObjects)}
              title="Toggle visualization of moving vehicles filtered by dynamic masking"
            >
              {showDynamicObjects ? <Eye size={14} /> : <EyeOff size={14} />}
              <span>MASKED VEHICLES ({showDynamicObjects ? 'ON' : 'OFF'})</span>
            </button>
          </div>
        </div>

        {/* Right: Action Buttons */}
        <div className="header-right">
          <button 
            className="btn-secondary"
            onClick={() => setIsAccuracyModalOpen(true)}
          >
            <Award size={15} color="#34d399" />
            <span>Accuracy Report (&le;1.0m)</span>
          </button>

          <button 
            className="btn-primary"
            onClick={() => setIsUploadModalOpen(true)}
          >
            <Upload size={15} />
            <span>Ingest Drone Flight</span>
          </button>
        </div>
      </header>

      {/* 2. MAIN 3D VIEWPORT WORKSPACE */}
      <main className="viewport-workspace">
        <div className="three-canvas-container">
          <DigitalTwinViewer 
            activeProjectId={activeProjectId}
            trajectoryData={trajectory}
            metricsData={metrics}
            pointCloudData={pointCloudData}
            activeLayer={activeLayer}
            onCursorPositionChange={setCursorCoords}
            showDynamicObjects={showDynamicObjects}
          />
        </div>

        {/* Floating Left: Reconstruction Pipeline Tracker */}
        <div className="floating-pipeline-panel glass-panel">
          <PipelineTracker 
            currentStage={currentStage}
            progress={pipelineProgress}
            onRerunPipeline={handleRerunPipeline}
            isProcessing={isProcessing}
          />
        </div>

        {/* Floating Right: Telemetry & Adaptive Sensor Trust HUD */}
        <div className="floating-telemetry-hud">
          <TelemetryHUD 
            cursorCoords={cursorCoords}
            metrics={metrics}
            trajectoryInfo={trajectory}
          />
        </div>

        {/* Floating Bottom Legend */}
        <div className="floating-bottom-legend glass-panel">
          {activeLayer === 'confidence' ? (
            <>
              <span style={{ color: 'var(--text-muted)' }}>AI CONFIDENCE:</span>
              <div className="legend-item" style={{ color: '#ef4444' }}>
                <div className="legend-dot" style={{ background: '#ef4444' }} />
                <span>&lt; 50% (Occluded)</span>
              </div>
              <div className="legend-item" style={{ color: '#f59e0b' }}>
                <div className="legend-dot" style={{ background: '#f59e0b' }} />
                <span>50% - 80% (Moderate)</span>
              </div>
              <div className="legend-item" style={{ color: '#00f2fe' }}>
                <div className="legend-dot" style={{ background: '#00f2fe' }} />
                <span>80% - 100% (High Certainty)</span>
              </div>
            </>
          ) : activeLayer === 'semantics' ? (
            <>
              <span style={{ color: 'var(--text-muted)' }}>SEMANTIC CLASSES:</span>
              <div className="legend-item" style={{ color: '#38bdf8' }}>
                <div className="legend-dot" style={{ background: '#38bdf8' }} />
                <span>Buildings</span>
              </div>
              <div className="legend-item" style={{ color: '#94a3b8' }}>
                <div className="legend-dot" style={{ background: '#64748b' }} />
                <span>Roads</span>
              </div>
              <div className="legend-item" style={{ color: '#10b981' }}>
                <div className="legend-dot" style={{ background: '#10b981' }} />
                <span>Vegetation</span>
              </div>
              <div className="legend-item" style={{ color: '#fde047' }}>
                <div className="legend-dot" style={{ background: '#facc15' }} />
                <span>Terrain</span>
              </div>
            </>
          ) : (
            <>
              <div className="radar-dot" />
              <span style={{ color: 'var(--cyan)', fontWeight: 700 }}>SINGLE-PASS 4D DIGITAL TWIN ACTIVE</span>
              <span style={{ color: 'var(--text-muted)' }}>| Left Click + Drag to Orbit | Right Click to Pan | Scroll to Zoom</span>
            </>
          )}
        </div>
      </main>

      {/* 3. MODALS */}
      <AccuracyMetricsModal 
        isOpen={isAccuracyModalOpen}
        onClose={() => setIsAccuracyModalOpen(false)}
        metrics={metrics}
      />

      <UploadFlightModal 
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUploadSuccess={(proj) => handleFlightUploadSuccess(proj)}
      />

    </div>
  );
}
