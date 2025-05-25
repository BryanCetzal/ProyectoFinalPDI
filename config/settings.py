import os

# Configuración de detección
DETECTION_CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Buscar el archivo YOLO en diferentes ubicaciones
POSSIBLE_YOLO_PATHS = [
    "data/yolov8n.pt",
    "data/yolov8s.pt", 
    "data/yolov8m.pt",
    "data/yolov8l.pt",
    "data/yolov8x.pt",
    "yolov8n.pt",  # En caso de que esté en la raíz
]

# Encontrar el primer archivo YOLO disponible
YOLO_MODEL_PATH = None
for path in POSSIBLE_YOLO_PATHS:
    if os.path.exists(path):
        YOLO_MODEL_PATH = path
        break

# Si no se encuentra, usar el nombre por defecto (se descargará automáticamente)
if YOLO_MODEL_PATH is None:
    YOLO_MODEL_PATH = "data/yolov8n.pt"

# Configuración de cámara
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Configuración de UI
WINDOW_TITLE = "Sistema de Detección y Conteo de Personas"
UI_UPDATE_INTERVAL = 50  # ms

# Rutas de archivos
DATA_DIR = "data"
OUTPUT_DIR = "output"
LOGS_DIR = "logs"

# Crear directorios si no existen
for directory in [DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

print(f"Configuración cargada - Modelo YOLO: {YOLO_MODEL_PATH}")