from ultralytics import YOLO

# 1. Load your trained custom model
model = YOLO("best.pt") 

# 2. Export to NCNN or TFLite (best for Raspberry Pi)
# We apply INT8 quantization to shrink the model size and speed up inference
model.export(format="onnx") 

# Alternatively, export to ONNX
# model.export(format="onnx", simplify=True)