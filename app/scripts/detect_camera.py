"""
Fast camera capture and display script with real-time face detection.
Uses OpenCV for optimal performance with inter-frame processing capability.
"""
import os
import cv2
from src.model import load_cascade, CascadeClassifier
from src.config import Configuration

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
    print(f"Camera resolution set to: {actual_width}x{actual_height} (16:9)")
    
    window_name = 'Camera - Face Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CONFIG.camera_window_width, CONFIG.camera_window_height)
    
    frame_count = 0
    print("Camera started. Press 'q' or ESC to quit.")
    
    while True:
        rval, frame = vc.read()
        
        if not rval:
            print("ERROR: Failed to read frame")
            break
        
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
        
        # ========================== Detect faces ==========================
        # print(f"Frame {frame_count}: Detected {len(faces_scaled)} faces")

        # ========================== Draw faces ==========================
        frame_with_boxes = draw_boxes(frame, faces_scaled)
        cv2.putText(
            frame_with_boxes,
            f'Frame: {frame_count} | Faces: {len(faces_scaled)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # ========================== Display frame ==========================
        cv2.imshow(window_name, frame_with_boxes)
        
        # ========================== Process keyboard input ==========================
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            print(f"Exiting. Processed {frame_count} frames.")
            break
        
        frame_count += 1
    
    # Cleanup
    vc.release()
    cv2.destroyAllWindows()
