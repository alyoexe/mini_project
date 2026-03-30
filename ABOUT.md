# About This Project

This project is a **Real-Time Object Detection and Explainability Pipeline** that combines computer vision object tracking with interpretable machine learning explanations.

## Project Purpose

The goal is to detect objects in video streams, track them across frames, identify uncertain predictions, and automatically generate visual explanations for why the model is uncertain about those detections using **LIME (Local Interpretable Model-agnostic Explanations)**.

This is useful for:
- **Model Debugging**: Understanding failure cases and edge cases.
- **Explainable AI**: Providing transparency on confident vs. uncertain predictions.
- **Quality Assurance**: Flagging objects that need manual review.
- **Dataset Improvement**: Collecting feedback on uncertain detections to improve training data.

## What the Project Does (Core Workflow)

1. **Ingests Video**: Reads a video file frame-by-frame using OpenCV.
2. **Runs Detection & Tracking**: Uses a YOLO model to detect objects and ByteTrack to maintain object IDs across frames.
3. **Identifies Uncertainty**: Flags detections with confidence scores in a configurable "uncertain" range (e.g., 0.40–0.75).
4. **Generates Explanations Asynchronously**: For uncertain objects, spawns background worker processes to compute LIME explanations without blocking the main pipeline.
5. **Visualizes Results**: Shows:
   - **Main window**: Bounding boxes with confidence scores, uncertainty labels, and in-frame LIME overlays.
   - **LIME window**: Tiled panel showing explanation heatmaps with fade-out timing.
6. **Provides Interactive Controls**: Pause, resume, toggle windows, and adjust settings in real-time.
7. **Profiles Performance**: Tracks FPS, tracking latency, LIME throughput, and memory usage.

## Key Technologies Used

### Core Libraries

| Library | Role |
|---------|------|
| **YOLO (ultralytics)** | Object detection model (ONNX format). Provides class predictions and confidence scores. |
| **ByteTrack** | Multi-object tracking. Assigns persistent IDs to objects across frames. |
| **LIME (lime)** | Generates local interpretable explanations by perturbing image regions and measuring prediction sensitivity. |
| **OpenCV (cv2)** | Video I/O, image processing, drawing annotations, window management. |
| **NumPy** | Numerical operations, array manipulation, masking. |
| **scikit-image** | Image segmentation (SLIC superpixels) for LIME feature generation. |
| **psutil** | System resource monitoring (memory profiling). |
| **multiprocessing** | Parallel worker processes for asynchronous LIME computation. |
| **tracemalloc** | Memory profiling and leak detection (optional). |

### Model Architecture

- **Detection Model**: YOLO (pretrained on large datasets like COCO).
  - Input: Video frames (variable resolution, downscalable).
  - Output: Bounding boxes, class probabilities, confidence scores.
  - Format: `best.onnx` (ONNX for cross-platform inference).

- **Tracking**: ByteTrack algorithm (lightweight, real-time).
  - Maintains object identities without expensive reidentification.

- **Explainability**: LIME algorithm.
  - Creates synthetic perturbed versions of the detected object crop.
  - Queries the YOLO model on perturbed images.
  - Learns a linear approximation of feature importance.
  - Outputs visual heatmaps showing which regions support/oppose the prediction.

### Hardware & Performance Considerations

- **Frame Processing**: Can run on CPU or GPU (ONNX inference agnostic).
- **Parallelism**: Multiprocessing spawns worker processes to avoid blocking main video loop.
- **Optimization Options**:
  - Frame skipping.
  - Tracking frequency reduction.
  - LIME quality modes (fast/balanced/high).
  - Batch processing of LIME jobs.
  - Crop caching to avoid redundant resizing.
  - Memory profiling to detect leaks.

## Project Structure

```
project/
├── exp7.py                          # Main pipeline script
├── best.onnx                        # Pretrained YOLO detection model
├── best.pt                          # PyTorch checkpoint (backup)
├── test_vid.mp4                     # Input video (user-provided)
├── README.md                        # Quick start and configuration guide
├── ABOUT.md                         # This file: project overview
├── feedback_data/
│   ├── feedback_log.jsonl           # Historical feedback records
│   └── crops/                       # Saved uncertain object crops
├── exp1.py - exp10.py               # Experimental variants
└── prg*.py, prepare_feedback_dataset.py # Utility scripts
```

## Config & Tuning Features

The pipeline is **highly configurable** via constants in `exp7.py`:

### Uncertainty Detection
```python
LOWER_THRESH, UPPER_THRESH  # Confidence range to flag as uncertain
UNCERTAIN_TIME              # How long an object must be uncertain before LIME
UNCERTAIN_STREAK_FRAMES     # Min frames in uncertain state
LIME_COOLDOWN              # Prevent re-explaining same object too often
```

### LIME Computation
```python
LIME_QUALITY_MODE          # "fast", "balanced", "high"
NUM_LIME_SAMPLES           # Perturbation count (higher = better but slower)
NUM_LIME_WORKERS           # Parallel explanation workers
ENABLE_BATCH_LIME          # Batch multiple objects together
MAX_LIME_JOBS_PER_SEC      # Global rate limit
```

### Performance
```python
FRAME_SKIP                 # Skip frames to reduce load
TRACK_EVERY_N_FRAMES       # Run detector less often
INFER_SCALE                # Downscale input for speed
ENABLE_CROP_CACHE          # Cache resized crops
```

### Visualization
```python
ENABLE_INFRAME_LIME        # Blend explanations on boxes
SHOW_LIME_WINDOW           # Show tiled explanation panel
SHOW_PERF_BREAKDOWN        # Display latency metrics
ENABLE_MASK_OVERLAY        # Render explanation heatmaps
```

## Data Flow

```
Video Input (test_vid.mp4)
    ↓
Read Frame (OpenCV)
    ↓
Resize & Inference (YOLO Model, ONNX)
    ↓
Track Objects (ByteTrack)
    ↓
[For each detected object]
    ├─ Confidence in "uncertain" range? → Queue LIME job
    ├─ Render Bounding Box (green=confident, red=uncertain)
    └─ Update UI
    ↓
[Async Worker Processes]
    ├─ Receive crop + object ID
    ├─ Generate LIME explanation (perturb + predict)
    ├─ Render heatmap overlay
    └─ Send overlay back to main process
    ↓
[Main Process]
    ├─ Receive overlay → store in memory
    ├─ Blend onto frame (in-frame window)
    ├─ Add to tiled panel (LIME window)
    └─ Fade out after hold duration
    ↓
Display Windows + FPS/Metrics
    ↓
User Controls (q/p/l/m)
```

## Use Cases

1. **Model Validation**: Quickly spot dataset biases or model edge cases.
2. **Iterative Improvement**: Collect uncertain predictions for manual annotation and retraining.
3. **Deployment Monitoring**: Monitor live detection streams for confidence anomalies.
4. **Education**: Teach students how object detection and explainability work together.
5. **Safety-Critical Systems**: Validate detector behavior before deployment in autonomous systems.

## Performance Characteristics

- **Real-time on modern hardware** (GPU recommended for < 30ms LIME per frame).
- **Scales with parallelism**: More workers = faster explanation throughput.
- **Memory overhead**: LIME buffers + cache; scales with `MAX_PENDING_JOBS` and cache size.
- **Configurable latency vs. quality**: Fast mode sacrifices explanation detail for speed.

## Extensions & Future Directions

- **Multi-model ensemble**: Combine explanations from multiple detectors.
- **Feedback loop**: Auto-retrain on collected uncertain samples.
- **Geographic heatmaps**: Aggregate uncertainty patterns across video.
- **API endpoint**: Wrap pipeline as REST service for integration.
- **Mobile deployment**: Strip multiprocessing overhead for on-device inference.

## Getting Started

1. Install dependencies: `pip install ultralytics opencv-python numpy lime scikit-image psutil`
2. Place model at `best.onnx` and video at `test_vid.mp4`.
3. Run: `python exp7.py`
4. See [README.md](README.md) for configuration details and troubleshooting.
