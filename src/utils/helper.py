import cv2
import numpy as np
from datetime import datetime

def calculate_centroid(bbox):
    """Calcula el centroide de una bounding box."""
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def calculate_distance(point1, point2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def draw_info_panel(frame, count, fps=0):
    """Dibuja panel de información en el frame."""
    height, width = frame.shape[:2]
    
    # Panel de información
    cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (300, 100), (255, 255, 255), 2)
    
    # Texto
    cv2.putText(frame, f'Personas detectadas: {count}', (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'FPS: {fps:.1f}', (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Timestamp: {datetime.now().strftime("%H:%M:%S")}', 
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def save_frame_with_timestamp(frame, output_dir="output"):
    """Guarda un frame con timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"frame_{timestamp}.png"
    filepath = f"{output_dir}/{filename}"
    cv2.imwrite(filepath, frame)
    return filepath