import threading

# One process-wide lock protects the free-tier server from loading/running the
# legacy video, AI Hub ONNX, Pose, Hand, and optional object models concurrently.
heavy_inference_lock = threading.RLock()
