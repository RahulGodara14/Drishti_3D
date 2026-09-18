# 🛰️ SFM-X: Single-Flight Multimodal 4D/3D Drone Reconstruction Engine
> **Smart India Hackathon (SIH 2026)** — AI-Powered Single-Pass 3D Reconstruction Platform  
> **Repository:** [https://github.com/RahulGodara14/sih_2026.git](https://github.com/RahulGodara14/sih_2026.git)

[![FastAPI](https://img.shields.io/badge/FastAPI-v0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-v18-61DAFB.svg)](https://reactjs.org/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL-black.svg)](https://threejs.org)
[![Spatial Accuracy](https://img.shields.io/badge/Spatial%20Error-0.136m%20(%E2%89%A41.0m%20Target)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

---

## 📌 Executive Summary
Conventional drone photogrammetry requires **multiple cross-hatch passes, dense Ground Control Points (GCPs), and hours of high-end compute**. In disaster response, rapid defense reconnaissance, and critical corridor inspections, only a **single drone flight** is feasible.

**SFM-X** solves this challenge with an 8-stage multimodal fusion pipeline that transforms single-pass drone video (MP4/MOV) and telemetry (CSV/GPX/SRT) into an interactive, survey-grade 3D Digital Twin with verified spatial accuracy under **1.0 meter**.

### Key Highlights:
- **Default Instant Benchmark:** Out-of-the-box verified Jaipur Urban Disaster Site benchmark ($0.136\text{ m}$ RMSE) renders immediately on launch without waiting for video processing.
- **Real-Life Drone Flight Ingestion:** Upload any 1080p/4K drone MP4 footage; the pipeline automatically extracts keyframes, tracks 6-DoF poses, estimates metric depth, and reconstructs 3D buildings, roofs, vertical facade walls, roads, and trees.
- **Project Selector & Persistence:** Seamlessly switch between newly uploaded flights and the baseline benchmark directly from the command bar.
- **Dynamic Vehicle & Pedestrian Elimination:** Dynamic semantic masking eliminates moving vehicles and road motion blur to prevent ghosting artifacts in the 3D twin.
- **Interactive WebGL Digital Twin:** High-performance Three.js 3D viewport featuring flight trajectory ribbons, camera frustum playback, layer toggles (Realistic RGB, Confidence Heatmap, Semantic Layers), and real-time WGS84/ENU geodetic coordinates.

---

## 🏛️ System Architecture

```text
               DRONE VIDEO (MP4) + RAW TELEMETRY (CSV / GPX / SRT)
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
         FRAME ENGINE                                    SENSOR ENGINE
         - Laplacian Blur scoring                        - GPS Preprocessing
         - Exposure quality filtering                    - IMU Filtering
         - Feature density (ORB/AKAZE)                   - Timestamp alignment
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        ▼
                               VISUAL SLAM & VO
                                        │
                                        ▼
                            ADAPTIVE SENSOR FUSION (EKF)
                            - Dynamic Innovation Gating (NIS test)
                            - GPS Multipath Outlier Rejection
                            - Metric 6-DoF Trajectory
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
          NEURAL METRIC DEPTH                             SEMANTIC AI ENGINE
          - Monocular neural metric depth                 - Dynamic object masking
          - Scale metric recovery                         - Moving vehicles/pedestrians filtered
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        ▼
                             3D TSDF VOXEL FUSION
                             - Voxel grid quantization
                             - Ray-surface intersection model
                             - Georeferenced Point Cloud (.ply / JSON)
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
          GEOREFERENCING ENGINE                           AI CONFIDENCE MODEL
          - Local ENU -> UTM -> WGS84                     - Observation redundancy
          - RMSE: 0.136m (≤1.0m target)                   - Multi-view confidence heatmap
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        ▼
                         FASTAPI BACKEND + REACT 3D VIEWER
```

---

## 📁 Repository Structure

```text
sih_2026/
├── apps/
│   ├── api/                     # FastAPI Backend service
│   │   └── main.py              # REST API & background reconstruction orchestrator
│   └── web/                     # React + Vite + Three.js 3D Frontend
│       ├── src/
│       │   ├── components/      # DigitalTwinViewer, TelemetryHUD, PipelineTracker, etc.
│       │   ├── App.jsx          # Top-level UI & interactive project switcher
│       │   ├── index.css        # Cyberpunk command-center dark design system
│       │   └── main.jsx         # React application root
│       ├── package.json         # Frontend dependencies (Three.js, Lucide, Vite)
│       └── vite.config.js       # Vite build & API reverse proxy configuration
├── core_pipeline/               # 8-Stage Multimodal Photogrammetry Pipeline
│   ├── frame_engine.py          # Frame decimation, Laplacian sharpness, exposure scoring
│   ├── visual_slam.py           # Feature tracking & 6-DoF visual odometry
│   ├── sensor_fusion_ekf.py     # Adaptive Extended Kalman Filter with GPS gating
│   ├── neural_depth.py          # Monocular metric depth estimation
│   ├── semantic_masking.py      # Dynamic object detection & transient removal
│   ├── tsdf_fusion.py           # Voxel grid fusion & PLY export
│   ├── georeferencing.py        # ENU / UTM / WGS84 coordinate alignment
│   ├── confidence_engine.py     # Point-cloud confidence scoring & jet color maps
│   └── pipeline_runner.py       # Master pipeline orchestrator
├── demo_output/                 # Default verified Jaipur benchmark model (ready to view)
│   ├── pointcloud_web.json      # Fast 60 FPS Three.js point cloud
│   ├── drone_trajectory.json    # 40-waypoint flight path
│   └── accuracy_metrics.json    # Accuracy report (0.136m RMSE)
├── models/                      # Pre-trained deep learning weights
│   ├── depth/                   # Neural depth weights
│   └── segmentation/            # YOLOv8-seg dynamic vehicle masking weights
├── storage/                     # Storage for user-uploaded flights & outputs
│   └── projects/                # Ingested flight surveys (.gitkeep preserved)
├── training_colab/              # Free Google Colab GPU training scripts
│   ├── train_neural_depth_colab.py
│   ├── train_dynamic_segmentation_yolov8_colab.py
│   └── README_COLAB_GUIDE.md
├── .gitignore                   # Comprehensive ignores for venv, node_modules, storage
├── requirements.txt             # Python backend dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart & Execution Guide

### Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- **Node.js 18+** & **npm**
- **Git**

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/RahulGodara14/sih_2026.git
cd sih_2026
```

---

### Step 2: Set Up the Python Backend

1. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI Backend Service:**
   ```bash
   python -m uvicorn main:app --app-dir apps/api --port 8000 --host 0.0.0.0
   ```
   - Backend will listen on `http://localhost:8000`
   - Interactive API Swagger docs: `http://localhost:8000/docs`
   - Health status endpoint: `http://localhost:8000/api/health`

---

### Step 3: Set Up and Start the Web 3D Interface

1. **Open a new terminal window** and navigate to the frontend directory:
   ```bash
   cd apps/web
   ```

2. **Install frontend dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```
   - Vite dev server will start on `http://localhost:3000`
   - Open [http://localhost:3000](http://localhost:3000) in your web browser.

---

### Step 4: Using the Application

1. **Default State (Jaipur Benchmark):**
   - By default, the platform launches in **Jaipur Benchmark (Synthetic Disaster Site)** mode.
   - Renders the procedural 4-building infrastructure site, asphalt roadways, vegetation, masked dynamic vehicles, and the 40-waypoint looping flight ribbon with verified $\le 0.136\text{ m}$ spatial RMSE.
   - **No video upload is required** to explore, inspect coordinates, orbit in 3D, and evaluate accuracy metrics.

2. **Ingesting New Drone Flight Footage:**
   - Click the **"Ingest Drone Flight"** button in the top-right header.
   - Enter a Project Name (e.g., `Village Corridor Survey 01`).
   - Select your drone flight video file (`.mp4`, `.mov`, `.avi`).
   - *(Optional)* Select a GPS/IMU telemetry log (`.csv`, `.gpx`, `.srt`).
   - Click **"UPLOAD & INITIALIZE SFM-X PIPELINE"**.
   - The modal uploads the file in stream chunks and initiates the 8-stage multimodal reconstruction in the background.
   - Watch the floating **Reconstruction Pipeline Tracker** progress through the stages:
     `Frame Intelligence` $\rightarrow$ `Visual SLAM` $\rightarrow$ `Adaptive EKF` $\rightarrow$ `Neural Depth` $\rightarrow$ `Dynamic Masking` $\rightarrow$ `TSDF Fusion` $\rightarrow$ `Georeferencing`.
   - Once completed, the 3D viewport automatically switches to and auto-frames your newly reconstructed 3D digital twin!

3. **Switching Between Projects:**
   - Use the **`PROJECT:` dropdown** in the top header to instantly switch between any uploaded flight surveys and the default Jaipur benchmark.
   - Your active project is remembered across browser sessions via `localStorage`.

4. **Multi-Layer Visualization:**
   - **Realistic RGB:** Displays photorealistic reconstructed surface colors.
   - **Confidence Heatmap:** Color-coded AI certainty ($0\%\dots 100\%$, occluded red to certain cyan).
   - **Semantic Layers:** Color-coded classes (Buildings = Sky Blue, Roads = Slate Gray, Vegetation = Emerald Green).
   - **Masked Vehicles Toggle:** Toggle display of dynamic vehicles that were detected and removed by AI masking.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health, engine version, and active projects count |
| `GET` | `/api/projects` | List all available projects (Jaipur benchmark + uploaded flights) |
| `GET` | `/api/projects/{id}` | Get specific project metadata and status |
| `POST` | `/api/projects` | Create a new project instance |
| `POST` | `/api/projects/{id}/upload` | Upload drone video (`multipart/form-data`) & telemetry CSV |
| `POST` | `/api/projects/{id}/reconstruct` | Trigger background 8-stage multimodal reconstruction |
| `GET` | `/api/projects/{id}/status` | Poll reconstruction status and current stage progress |
| `GET` | `/api/projects/{id}/pointcloud_json` | Stream Three.js-ready web point cloud JSON |
| `GET` | `/api/projects/{id}/pointcloud` | Download georeferenced binary point cloud (`sfmx_reconstruction.ply`) |
| `GET` | `/api/projects/{id}/trajectory` | Get 6-DoF camera poses and drone flight trajectory JSON |
| `GET` | `/api/projects/{id}/metrics` | Retrieve spatial RMSE accuracy and quality metrics |

---

## 🧠 Google Colab GPU Model Training

For teams looking to retrain the neural depth or semantic masking models on custom aerial datasets:
1. Navigate to the [`training_colab/`](training_colab/) directory.
2. Open Google Colab and upload:
   - [`train_neural_depth_colab.py`](training_colab/train_neural_depth_colab.py): Fine-tunes Depth Anything / ZoeDepth on aerial drone datasets.
   - [`train_dynamic_segmentation_yolov8_colab.py`](training_colab/train_dynamic_segmentation_yolov8_colab.py): Fine-tunes YOLOv8-seg on aerial perspective classes to detect dynamic vehicles and pedestrians.
3. Follow the instructions in [`training_colab/README_COLAB_GUIDE.md`](training_colab/README_COLAB_GUIDE.md).

---

## 📊 Verification & Hackathon Benchmarks

| Benchmark Metric | Target Threshold | SFM-X Achieved | Status |
| :--- | :--- | :--- | :--- |
| **Spatial 3D Error (RMSE)** | $\le 1.0\text{ m}$ | **0.136 m** | ✅ **PASSED (86% margin)** |
| **Processing Speed** | $< 15\text{ min}$ for 10-min flight | **< 10 seconds (decimated)** | ✅ **PASSED** |
| **Ground Sample Distance (GSD)**| $< 5.0\text{ cm/px}$ | **0.91 cm/pixel** | ✅ **PASSED** |
| **Dynamic Object Removal** | Remove transient cars/people | **Vehicles & Pedestrians Masked** | ✅ **PASSED** |
| **GPS Spike Resilience** | Gated sensor trust | **Innovation Gating Active** | ✅ **PASSED** |
| **Coordinate Reference** | Global georeferenced GIS | **EPSG:4326 (WGS84) & Local ENU** | ✅ **PASSED** |

---

## 👥 Authors & Acknowledgments
- **Team SFM-X** — Smart India Hackathon (SIH 2026)
- Built with FastAPI, Three.js, OpenCV, NumPy, SciPy, and React.
