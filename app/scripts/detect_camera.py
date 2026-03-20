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

    main(CONFIG)

def draw_boxes(img, faces):
    """Draw bounding boxes on detected faces"""
    for crop in faces:
        x, y, w, h = crop['x'], crop['y'], crop['w'], crop['h']

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return img


def main(CONFIG):
    """Capture and display camera frames with face detection"""
    cascade_path = os.path.join(CONFIG.haar_cascades, 'haarcascade_frontalface_default.xml')
    cascade = load_cascade(cascade_path)
    CONFIG.crop_size = max(cascade.height, cascade.width)
    classifier = CascadeClassifier(CONFIG, cascade)

    
    # Open video capture
    vc = cv2.VideoCapture(0)
    
    if not vc.isOpened():
        print("ERROR: Cannot open video capture device")
        return
    
    # Optional: Set camera resolution for better performance
    # vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    # vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    vc.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    print("Camera started. Press 'q' or ESC to quit.")
    
    while True:
        rval, frame = vc.read()
        
        if not rval:
            print("ERROR: Failed to read frame")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ========================== Detect faces ==========================
        faces = classifier.predict(img=gray)
        print(f"Frame {frame_count}: Detected {len(faces)} faces")

        # ========================== Draw faces ==========================
        frame_with_boxes = draw_boxes(frame, faces)
        cv2.putText(
            frame_with_boxes,
            f'Frame: {frame_count} | Faces: {len(faces)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # ========================== Display frame ==========================
        cv2.imshow('Camera - Face Detection', frame_with_boxes)
        
        # ========================== Process keyboard input ==========================
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            print(f"Exiting. Processed {frame_count} frames.")
            break
        
        frame_count += 1
    
    # Cleanup
    vc.release()
    cv2.destroyAllWindows()
