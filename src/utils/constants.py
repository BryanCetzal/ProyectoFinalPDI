# Clases YOLO (índice 0 = persona)
PERSON_CLASS_ID = 0

# Colores para visualización (BGR)
COLORS = {
    'BOUNDING_BOX': (0, 0, 255),      # Rojo
    'CENTROID': (0, 255, 0),          # Verde
    'TEXT': (255, 255, 255),          # Blanco
    'BACKGROUND': (0, 0, 0)           # Negro
}

# Configuración de tracking
MAX_TRACKING_DISTANCE = 50
MIN_TRACKING_FRAMES = 5