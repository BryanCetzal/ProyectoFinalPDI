import os
from ultralytics import YOLO

class ModelLoader:
    def __init__(self, model_path="data/yolov8n.pt"):
        self.model_path = model_path
        self.model = None
    
    def load_model(self):
        """Carga el modelo YOLO."""
        try:
            if not os.path.exists(self.model_path):
                print(f"Descargando modelo YOLO a {self.model_path}...")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            self.model = YOLO(self.model_path)
            print(f"Modelo YOLO cargado exitosamente: {self.model_path}")
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def get_model(self):
        """Obtiene el modelo cargado."""
        return self.model