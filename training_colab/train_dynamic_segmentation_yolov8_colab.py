# -*- coding: utf-8 -*-
"""
Google Colab Training Script: Dynamic Object Segmentation for Aerial Drone Imagery (YOLOv8-seg)
Run this script in Google Colab with GPU.
Fine-tunes YOLOv8-seg on drone overhead datasets (VisDrone / UAVDT) to extract precise
dynamic masks for moving vehicles, trucks, pedestrians, and bikes.

Instructions in Colab:
1. Runtime -> Change runtime type -> Select GPU (T4/A100).
2. Run: !pip install ultralytics roboflow opencv-python -q
3. Run this script!
"""

from ultralytics import YOLO
import os

print("="*60)
print("SFM-X Aerial Dynamic Object Segmentation Training (YOLOv8-seg)")
print("="*60)

def train_yolo_segmentation():
    # Load pretrained YOLOv8-seg model (Nano for fast drone inference, Medium/X for max accuracy)
    model = YOLO("yolov8n-seg.pt")
    
    # In Colab, you can directly download the VisDrone-DET / VisDrone-VID dataset:
    # Example dataset yaml: 'VisDrone.yaml' (built-in in ultralytics) or custom drone dataset
    print("Beginning fine-tuning on aerial dataset...")
    
    results = model.train(
        data="coco8-seg.yaml",    # Built-in ultralytics mini-benchmark; replace with VisDrone-seg.yaml
        epochs=15,
        imgsz=640,
        batch=16,
        device=0,                 # GPU in Colab
        workers=4,
        project="sfmx_drone_seg",
        name="aerial_dynamic_masker",
        save=True,
        save_period=5,
        # Focus on dynamic classes: person (0), bicycle (1), car (2), motorcycle (3), bus (5), truck (7)
        classes=[0, 1, 2, 3, 5, 7]
    )
    
    # Retrieve exact save directory from Ultralytics results
    save_dir = getattr(results, "save_dir", "runs/segment/sfmx_drone_seg/aerial_dynamic_masker")
    best_weights = os.path.join(save_dir, "weights", "best.pt")
    
    # Export trained model to ONNX
    exported_path = model.export(format="onnx")
    
    print("\n" + "="*60)
    print(f"🎉 Training completed successfully!")
    print(f"📍 Original Weights location: {best_weights}")
    
    # Auto-copy to /content/ root so it appears directly in Colab's main file sidebar
    import shutil
    root_pt = "/content/best.pt" if os.path.exists("/content") else "best.pt"
    root_onnx = "/content/best.onnx" if os.path.exists("/content") else "best.onnx"
    
    if os.path.exists(best_weights):
        shutil.copy(best_weights, root_pt)
        print(f"✅ Copied to root: {root_pt} (check your Colab files sidebar!)")
        
    if exported_path and os.path.exists(exported_path):
        shutil.copy(exported_path, root_onnx)
        print(f"✅ Copied to root: {root_onnx}")
        
    print("="*60)
    print("👉 To download directly in Colab, run:")
    print("from google.colab import files")
    print("files.download('/content/best.pt')")
    print("="*60)

if __name__ == "__main__":
    train_yolo_segmentation()
