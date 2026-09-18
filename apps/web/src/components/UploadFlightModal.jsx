import React, { useState } from 'react';
import { X, Upload, Video, FileSpreadsheet, Loader2 } from 'lucide-react';

export default function UploadFlightModal({ isOpen, onClose, onUploadSuccess }) {
  const [videoFile, setVideoFile] = useState(null);
  const [telemetryFile, setTelemetryFile] = useState(null);
  const [projectName, setProjectName] = useState('New Drone Flight Survey');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [errorMessage, setErrorMessage] = useState(null);

  if (!isOpen) return null;

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!videoFile) {
      setErrorMessage("Please select an MP4/MOV drone video file to ingest.");
      return;
    }

    setIsUploading(true);
    setErrorMessage(null);
    setUploadStatus('Creating survey project on backend...');

    try {
      // 1. Create project
      const createRes = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: projectName || `Flight: ${videoFile.name}` })
      });
      if (!createRes.ok) {
        throw new Error(`Failed to create project (HTTP ${createRes.status})`);
      }
      const project = await createRes.json();
      const pid = project.id;

      // 2. Upload actual video and telemetry files via multipart/form-data
      setUploadStatus(`Uploading ${videoFile.name} (${(videoFile.size / (1024*1024)).toFixed(1)} MB)...`);
      const uploadForm = new FormData();
      uploadForm.append('video', videoFile);
      if (telemetryFile) uploadForm.append('telemetry_csv', telemetryFile);

      const uploadRes = await fetch(`/api/projects/${pid}/upload`, {
        method: 'POST',
        body: uploadForm
      });
      if (!uploadRes.ok) {
        throw new Error(`File upload failed (HTTP ${uploadRes.status})`);
      }

      // 3. Trigger 8-stage multimodal reconstruction
      setUploadStatus('Initializing 8-stage SFM-X multimodal pipeline...');
      const reconRes = await fetch(`/api/projects/${pid}/reconstruct`, {
        method: 'POST'
      });
      if (!reconRes.ok) {
        throw new Error(`Reconstruction trigger failed (HTTP ${reconRes.status})`);
      }
      
      setUploadStatus('Flight ingested successfully! Starting 3D digital twin...');
      setTimeout(() => {
        if (onUploadSuccess) {
          onUploadSuccess(project);
        }
        setIsUploading(false);
        onClose();
      }, 500);

    } catch (err) {
      console.error("Upload error:", err);
      setErrorMessage(err.message || 'Could not connect to FastAPI backend on port 8000. Ensure server is online.');
      setIsUploading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '480px' }}>
        <div className="modal-header">
          <div className="modal-title-group">
            <div style={{ padding: '8px', borderRadius: '8px', background: 'rgba(0, 242, 254, 0.15)', color: '#00f2fe' }}>
              <Upload size={18} />
            </div>
            <div>
              <div className="modal-title">Ingest Drone Flight Footage</div>
              <div className="modal-subtitle">Upload 4K/1080p Video + GPS/IMU Telemetry</div>
            </div>
          </div>
          <button className="modal-btn-close" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <form onSubmit={handleUpload} style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
          <div>
            <label style={{ display: 'block', color: 'var(--text-muted)', marginBottom: '5px' }}>PROJECT / SITE IDENTIFIER</label>
            <input 
              type="text" 
              style={{ width: '100%', background: '#0a1020', border: '1px solid var(--border-subtle)', borderRadius: '8px', padding: '9px 12px', color: '#fff', outline: 'none' }}
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              placeholder="e.g. Flood Zone Alpha - Flight 01"
              required
            />
          </div>

          <div>
            <label style={{ display: 'block', color: 'var(--text-muted)', marginBottom: '5px' }}>DRONE FLIGHT VIDEO (MP4 / MOV)</label>
            <label style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', border: '1px dashed var(--border-cyan)', borderRadius: '8px', padding: '16px', cursor: 'pointer', background: 'rgba(0, 242, 254, 0.02)' }}>
              <Video size={24} color="#00f2fe" style={{ marginBottom: '6px' }} />
              <span style={{ color: '#fff', fontWeight: 600 }}>{videoFile ? videoFile.name : 'Select Drone Video (1080p / 4K)'}</span>
              <span style={{ color: 'var(--text-muted)', fontSize: '10px', marginTop: '2px' }}>Supports DJI, Autel, Skydio MP4 streams</span>
              <input 
                type="file" 
                accept="video/*" 
                style={{ display: 'none' }} 
                onChange={(e) => setVideoFile(e.target.files[0])}
              />
            </label>
          </div>

          <div>
            <label style={{ display: 'block', color: 'var(--text-muted)', marginBottom: '5px' }}>TELEMETRY LOG (CSV / SRT / GPX)</label>
            <label style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', border: '1px dashed rgba(16, 185, 129, 0.4)', borderRadius: '8px', padding: '16px', cursor: 'pointer', background: 'rgba(16, 185, 129, 0.02)' }}>
              <FileSpreadsheet size={24} color="#34d399" style={{ marginBottom: '6px' }} />
              <span style={{ color: '#fff', fontWeight: 600 }}>{telemetryFile ? telemetryFile.name : 'Select GPS / IMU Telemetry'}</span>
              <span style={{ color: 'var(--text-muted)', fontSize: '10px', marginTop: '2px' }}>Timestamp, Lat, Lon, Altitude, Yaw, Pitch, Roll</span>
              <input 
                type="file" 
                accept=".csv,.srt,.gpx,.json" 
                style={{ display: 'none' }} 
                onChange={(e) => setTelemetryFile(e.target.files[0])}
              />
            </label>
          </div>

          {errorMessage && (
            <div style={{ padding: '10px 12px', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.4)', color: '#fca5a5', fontSize: '11px', lineHeight: 1.4 }}>
              <strong>Upload Error:</strong> {errorMessage}
            </div>
          )}

          {isUploading && uploadStatus && (
            <div style={{ padding: '8px 12px', borderRadius: '8px', background: 'rgba(0, 242, 254, 0.1)', border: '1px solid rgba(0, 242, 254, 0.3)', color: '#00f2fe', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Loader2 size={14} className="animate-spin" />
              <span>{uploadStatus}</span>
            </div>
          )}

          <div style={{ paddingTop: '8px' }}>
            <button 
              type="submit" 
              className="btn-primary"
              style={{ width: '100%', justifyContent: 'center', padding: '10px' }}
              disabled={isUploading}
            >
              {isUploading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span>INGESTING FLIGHT DATA...</span>
                </>
              ) : (
                <>
                  <Upload size={16} />
                  <span>UPLOAD & INITIALIZE SFM-X PIPELINE</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
