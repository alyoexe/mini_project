# Real-Time Detection with LIME Explanations

This project runs object detection and tracking on a video stream, then generates **LIME explanations** for uncertain detections.

The main entry point is [exp7.py](exp7.py).

## What This Script Does

- Loads a YOLO model from ONNX (`best.onnx` by default).
- Runs tracking with ByteTrack.
- Marks detections as uncertain when confidence is between thresholds.
- Triggers asynchronous LIME jobs for uncertain tracked objects.
- Shows:
	- A main detection window (`Detection`)
	- An optional tiled explanation window (`LIME`)
- Supports performance tuning (frame skip, tracking rate, batching, worker count).

## Files in This Folder

- [exp7.py](exp7.py): main pipeline script.
- `best.onnx`: detection model used at runtime.
- `best.pt`: training/checkpoint artifact (not used directly by `exp7.py` runtime).
- [feedback_data/feedback_log.jsonl](feedback_data/feedback_log.jsonl): feedback log data.
- [feedback_data/crops/](feedback_data/crops/): saved crop directory.

## Requirements

Install Python 3.9+ and these packages:

```bash
pip install ultralytics opencv-python numpy lime scikit-image psutil
```

Notes:
- `multiprocessing` and `tracemalloc` are standard library modules.
- On Windows, `spawn` mode is already handled in the script.

## Quick Start

1. Put your model at `best.onnx` (or change `MODEL_PATH` in [exp7.py](exp7.py#L23)).
2. Put your video file at `test_vid.mp4` (or change `VIDEO_SOURCE` in [exp7.py](exp7.py#L24)).
3. Run:

```bash
python exp7.py
```

## Keyboard Controls

- `q`: quit
- `p`: pause/resume video
- `l`: toggle LIME tiled window
- `m`: toggle in-frame LIME overlay

## Core Configuration (exp7.py)

You can tune behavior by editing constants near the top of [exp7.py](exp7.py).

### Detection and Uncertainty

- `LOWER_THRESH`, `UPPER_THRESH`: confidence range considered uncertain.
- `UNCERTAIN_TIME`: minimum uncertain duration before explanation is allowed.
- `UNCERTAIN_STREAK_FRAMES`: minimum uncertain frame streak.
- `LIME_COOLDOWN`: per-object cooldown between explanations.

### LIME Runtime

- `NUM_LIME_SAMPLES`: perturbation samples per explanation.
- `LIME_QUALITY_MODE`: `fast`, `balanced`, or `high`.
- `ENABLE_BATCH_LIME`, `BATCH_LIME_MAX`, `BATCH_FLUSH_SEC`: queue batching behavior.
- `NUM_LIME_WORKERS`: parallel explanation workers.
- `MAX_LIME_JOBS_PER_SEC`: global explanation rate limiter.

### Performance

- `FRAME_SKIP`: skip incoming frames to reduce load.
- `TRACK_EVERY_N_FRAMES`: run tracker less frequently.
- `INFER_SCALE`: downscale input before tracking.
- `TRACK_IMGSZ`, `LIME_IMGSZ`: detector inference sizes.

### Visualization

- `ENABLE_INFRAME_LIME`: blend explanations on object boxes.
- `SHOW_LIME_WINDOW`: show tiled LIME panel.
- `ENABLE_MASK_OVERLAY`, `ENABLE_CONTOURS`, `ENABLE_MASK_SMOOTH`: mask rendering details.
- `SHOW_PERF_BREAKDOWN`: show tracking/LIME metrics overlay.

## How the Pipeline Works

1. Read frame.
2. Run tracking (`model.track(...)`) at configured cadence.
3. For each tracked object:
	 - If confidence is uncertain, accumulate uncertain state.
	 - If conditions pass, crop object and queue LIME job asynchronously.
4. Worker process runs LIME with YOLO prediction callback.
5. Main process receives overlays and stores explanation state.
6. Explanations are rendered in-frame and/or in a tiled panel with fade-out timing.

## Troubleshooting

- If no window appears:
	- Confirm OpenCV GUI support (`opencv-python`, not headless build).
	- Verify video path exists and can be opened.

- If the script is slow:
	- Set `LIME_QUALITY_MODE = "fast"`.
	- Increase `TRACK_EVERY_N_FRAMES` to `2` or `3`.
	- Increase `FRAME_SKIP`.
	- Disable overlays: `ENABLE_INFRAME_LIME = False`, `SHOW_LIME_WINDOW = False`.

- If LIME jobs back up:
	- Increase `MAX_LIME_JOBS_PER_SEC` carefully.
	- Increase `NUM_LIME_WORKERS` (watch CPU/GPU usage).
	- Reduce `NUM_LIME_SAMPLES`.

- If IDs disappear too quickly:
	- Adjust `LOST_TRACK_GRACE_SEC`, `EXPLANATION_HOLD_SEC`, and `EXPLANATION_FADE_SEC`.

## Suggested Starting Profiles

### Fastest

- `LIME_QUALITY_MODE = "fast"`
- `TRACK_EVERY_N_FRAMES = 2`
- `FRAME_SKIP = 1`
- `SHOW_LIME_WINDOW = False`
- `ENABLE_INFRAME_LIME = False`

### Balanced (default style)

- `LIME_QUALITY_MODE = "balanced"`
- `TRACK_EVERY_N_FRAMES = 1`
- `FRAME_SKIP = 0`

### Highest explanation detail

- `LIME_QUALITY_MODE = "high"`
- `NUM_LIME_SAMPLES >= 50`
- Keep frame skip low and use a stronger machine.

## Entry Point

The script starts from:

```python
if __name__ == "__main__":
		main()
```

So running `python exp7.py` is sufficient.
