import React from 'react';
import { 
  CheckCircle2, Loader2, Sparkles, Filter, Layers, 
  Map, Zap, Cpu, Clock, Check
} from 'lucide-react';

export default function PipelineTracker({ currentStage = 'completed', progress = 100, onRerunPipeline, isProcessing }) {
  const stages = [
    { id: 'frame_engine', name: 'Frame Intelligence', desc: 'Laplacian blur filter & 2-4 FPS decimation', icon: Filter, target: '18,000 -> 1,500 frames' },
    { id: 'visual_slam', name: 'Visual SLAM & VO', desc: 'Feature tracking & relative 6-DoF poses', icon: Zap, target: 'Sub-pixel accuracy' },
    { id: 'sensor_fusion', name: 'Adaptive EKF Fusion', desc: 'Visual pose + GPS/IMU innovation gating', icon: Cpu, target: 'GPS spike rejection' },
    { id: 'neural_depth', name: 'Neural Metric Depth', desc: 'Monocular depth + scale metric recovery', icon: Sparkles, target: 'Metric calibrated' },
    { id: 'dynamic_masking', name: 'Dynamic Object Masking', desc: 'AI vehicle & pedestrian removal', icon: Layers, target: '3 transient objects removed' },
    { id: 'tsdf_fusion', name: '3D TSDF Voxel Fusion', desc: 'RGB-D voxel integration & point cloud', icon: Layers, target: '18.5k points fused' },
    { id: 'georeferencing', name: 'WGS84 Georeferencing', desc: 'Local ENU -> UTM -> EPSG:4326', icon: Map, target: '<= 1.0m verified (0.136m)' }
  ];

  return (
    <>
      <div className="panel-header">
        <div className="panel-title-group">
          <Clock size={15} />
          <span>Reconstruction Pipeline</span>
        </div>
        <div className="panel-status-tag">
          <span>{progress.toFixed(0)}%</span>
          {isProcessing ? (
            <Loader2 size={13} className="animate-spin" color="#00f2fe" />
          ) : (
            <CheckCircle2 size={13} color="#34d399" />
          )}
        </div>
      </div>

      {/* Real-time Stage Sequence */}
      <div className="stage-list">
        {stages.map((stage, idx) => {
          const Icon = stage.icon;
          const isDone = progress >= ((idx + 1) / stages.length) * 95;
          const isCurrent = !isDone && progress >= (idx / stages.length) * 90;

          return (
            <div 
              key={stage.id} 
              className={`stage-item ${isCurrent ? 'current' : (isDone ? 'done' : '')}`}
            >
              <div className="stage-header">
                <div className="stage-info">
                  <div style={{ color: isDone ? '#34d399' : (isCurrent ? '#00f2fe' : '#64748b') }}>
                    <Icon size={14} />
                  </div>
                  <div>
                    <div className="stage-name">{stage.name}</div>
                    <div className="stage-desc">{stage.desc}</div>
                  </div>
                </div>
                {isDone ? (
                  <Check size={13} color="#34d399" />
                ) : isCurrent ? (
                  <Loader2 size={13} className="animate-spin" color="#00f2fe" />
                ) : null}
              </div>
              <div className="stage-benchmark">
                <span style={{ color: 'var(--text-muted)' }}>BENCHMARK:</span>
                <span>{stage.target}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Rerun / Processing Button */}
      <button 
        className="btn-run-pipeline"
        onClick={onRerunPipeline}
        disabled={isProcessing}
      >
        {isProcessing ? (
          <>
            <Loader2 size={14} className="animate-spin" />
            <span>PROCESSING MULTIMODAL FUSION...</span>
          </>
        ) : (
          <>
            <Zap size={14} />
            <span>RUN FAST RECONSTRUCTION</span>
          </>
        )}
      </button>
    </>
  );
}
