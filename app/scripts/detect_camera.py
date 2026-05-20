"""
Fast camera capture and display script with real-time face detection.
Uses OpenCV for optimal performance with inter-frame processing capability.
"""
import os
import cv2
import time
import threading
import numpy as np
from src.model import load_cascade, CascadeClassifier
from src.config import Configuration


class ThreadedCamera:
    """Continuously grabs frames in a background thread so vc.read() never blocks the main loop."""

    def __init__(self, vc: cv2.VideoCapture):
        self.vc = vc
        self.frame = None
        self.ret = False
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while not self._stopped:
            ret, frame = self.vc.read()
            with self._lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self._lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stopped = True
        self._thread.join()

def start_detect_camera(CONFIG: Configuration):
    # Disable Qt GUI backend if no display is available
    if not os.environ.get('DISPLAY'):
        cv2.setUseOptimized(True)
        # Force use of non-GUI backend for headless environments
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    camera(CONFIG)

def draw_boxes(img, faces):
    """Draw bounding boxes on detected faces"""
    for crop in faces:
        x, y, w, h = crop['x'], crop['y'], crop['w'], crop['h']

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return img


def camera(CONFIG: Configuration):
    """Capture and display camera frames with face detection"""
    # cascade_path = os.path.join(CONFIG.cv_haar_cascades, 'haarcascade_frontalface_default.xml')
    # cascade_path = os.path.join(CONFIG.computed_haar_cascades, 'haar_cascade_stage_6_fpr_0.8265.xml')
    # cascade_path = os.path.join(CONFIG.computed_haar_cascades, 'haar_cascade_stage_13_fpr_0.0000.xml')
    # cascade_path = os.path.join(CONFIG.computed_haar_cascades, 'haar_cascade_stage_21_fpr_0.0000.xml')
    # cascade_path = os.path.join(CONFIG.computed_haar_cascades, 'haar_cascade_stage_49_fpr_0.0000.xml')
    # cascade_path = os.path.join(CONFIG.computed_haar_cascades, 'haar_cascade_stage_13_fpr_0.0049.xml')
    # cascade_path = os.path.join(CONFIG.computed_haar_cascades, 'haar_cascade_stage_53_fpr_0.0001.xml')
    # cascade_path = os.path.join(CONFIG.computed_haar_cascades, 'haar_cascade_stage_38_fpr_0.0002.xml')
    # cascade_path = os.path.join(CONFIG.computed_haar_cascades, 'haar_cascade_stage_10_fpr_0.0043.xml')
    # cascade_path = os.path.join(CONFIG.computed_haar_cascades, 'haar_cascade_stage_12_fpr_0.0030.xml')
    cascade_path = CONFIG.computed_haar_cascades_path

    cascade = load_cascade(cascade_path)
    CONFIG.crop_size = max(cascade.height, cascade.width)
    classifier = CascadeClassifier(CONFIG, cascade)

    # Open video capture
    vc = cv2.VideoCapture(0)
    
    if not vc.isOpened():
        print("ERROR: Cannot open video capture device")
        return
    
    # Set camera to 16:9 resolution matching detect_width
    # detect_width is the width; height = width * 9 / 16 for 16:9 aspect ratio
    detect_width = CONFIG.detect_width
    detect_height = int(detect_width * 9 / 16)
    
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, detect_width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, detect_height)
    vc.set(cv2.CAP_PROP_FPS, 30)
    
    # Read actual resolution set by camera
    actual_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = vc.get(cv2.CAP_PROP_FPS)
    print(f"Camera resolution: {actual_width}x{actual_height} | Camera FPS: {actual_fps}")
    
    # Threaded capture — main loop is no longer blocked by camera framerate
    cam = ThreadedCamera(vc)

    window_name = 'Camera - Face Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CONFIG.camera_window_width, CONFIG.camera_window_height)
    
    frame_count = 0
    fps = 0.0
    fps_valid = False
    start_time = None
    fps_history = []
    print("Camera started. Press 'q' or ESC to quit.")
    
    while True:
        rval, frame = cam.read()
        
        if not rval or frame is None:
            continue
        
        # Start timing from the first actual frame
        if start_time is None:
            start_time = time.time()
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with halve_size downsampling
        faces = classifier.predict(img=gray, halve_size=True, halve_size_factor=CONFIG.halve_size_factor)
        
        # Scale face coordinates back to original frame size
        scale_factor = CONFIG.halve_size_factor
        faces_scaled = [
            {
                "x": int(f["x"] * scale_factor),
                "y": int(f["y"] * scale_factor),
                "w": int(f["w"] * scale_factor),
                "h": int(f["h"] * scale_factor),
            }
            for f in faces
        ]

        frame_with_boxes = draw_boxes(frame, faces_scaled)

        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            # Discard the first sample — includes setup/warmup overhead
            if fps_valid:
                fps_history.append(fps)
            fps_valid = True
            frame_count = 0
            start_time = time.time()

        # Compute running stats
        if fps_valid:
            arr = np.array(fps_history)
            avg_fps = arr.mean()
            std_fps = arr.std()
        else:
            avg_fps = std_fps = 0.0

        label = f'FPS: {fps:.1f} | Avg: {avg_fps:.1f} | Std: {std_fps:.2f} | Faces: {len(faces_scaled)}' if fps_valid else f'Warming up... | Faces: {len(faces_scaled)}'
        cv2.putText(
            frame_with_boxes,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(window_name, frame_with_boxes)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Print final stats
    if fps_history:
        arr = np.array(fps_history)
        print(f"\n--- FPS Stats ({len(arr)} samples) ---")
        print(f"  Avg: {arr.mean():.2f}")
        print(f"  Std: {arr.std():.2f}")
        print(f"  Min: {arr.min():.2f}")
        print(f"  Max: {arr.max():.2f}")
    else:
        print("No FPS samples collected.")

    cam.stop()
    vc.release()
    cv2.destroyAllWindows()
