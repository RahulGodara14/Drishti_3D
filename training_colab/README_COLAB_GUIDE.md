# 🚀 Google Colab Model Training Guide for SFM-X

This guide explains how to train or fine-tune the deep learning models for **SFM-X** using free Google Colab GPUs (T4 or A100), keeping your local storage completely clean on your E: drive.

---

## 1. Neural Metric Depth Model (`train_neural_depth_colab.py`)
- **Purpose**: Estimates dense per-pixel metric depth from single drone video frames.
- **Dataset options**:
  - [VisDrone Dataset](http://aiskyeye.com/)
  - [UAVDT (Unmanned Aerial Vehicle Benchmark)](https://sites.google.com/site/daviddo9512/uavdt)
  - [Depth Anything V2 Benchmark](https://github.com/DepthAnything/Depth-Anything-V2)
- **Colab Steps**:
  1. Go to [colab.research.google.com](https://colab.research.google.com/) and create a **New Notebook**.
  2. Select **Runtime** ➔ **Change runtime type** ➔ Choose **T4 GPU** (free) or **A100**.
  3. In a cell, run:
     ```bash
     !pip install torch torchvision transformers datasets accelerate -q
     ```
  4. Paste and run the code from [`train_neural_depth_colab.py`](train_neural_depth_colab.py).
  5. Once training completes, download `sfmx_depth_drone_model.pth` and place it in your local workspace under `models/depth/`.

---

## 2. Dynamic Object Segmentation (`train_dynamic_segmentation_yolov8_colab.py`)
- **Purpose**: Detects and segments moving vehicles (cars, trucks, buses, motorcycles) and pedestrians to generate dynamic rejection masks, preventing ghosting trails in the 3D reconstructed mesh and point cloud.
- **Colab Steps**:
  1. Open Google Colab with GPU enabled.
  2. In a cell, run:
     ```bash
     !pip install ultralytics roboflow opencv-python -q
     ```
  3. Paste and run the script from [`train_dynamic_segmentation_yolov8_colab.py`](train_dynamic_segmentation_yolov8_colab.py).
  4. **Where to find `best.pt` in Colab's file browser**:
     - Expand the **`runs`** folder in the left sidebar:
       📁 `runs` ➔ 📁 `segment` ➔ 📁 `sfmx_drone_seg` ➔ 📁 `aerial_dynamic_masker` ➔ 📁 `weights` ➔ 📄 **`best.pt`** & **`best.onnx`**
     - Or simply run this in a new Colab cell to download it instantly to your computer:
       ```python
       from google.colab import files
       files.download('/content/runs/segment/sfmx_drone_seg/aerial_dynamic_masker/weights/best.pt')
       ```
  5. Download `best.pt` and place it in your local workspace under `models/segmentation/`.

---

## 3. Deployment into SFM-X Local Pipeline
Once weights are placed in your `models/` directory, the local `core_pipeline/` will automatically detect and prioritize the trained model weights. When running on CPU or without custom weights, the system defaults to high-accuracy multi-scale gradient and geometric estimation so everything works seamlessly out of the box!
