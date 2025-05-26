import numpy as np
from src.utils.constants import PERSON_CLASS_ID
from src.utils.helper import calculate_centroid

class PersonDetector:
    def __init__(self, model, confidence_threshold=0.5):
        self.model = model
        self.confidence_threshold = confidence_threshold
    
    def detect_persons(self, frame):
        """Detecta personas en el frame."""
        results = self.model(frame)
        result = results[0]
        
        # Extraer información de detecciones
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        confidence = np.array(result.boxes.conf.cpu())
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        
        # Filtrar solo personas con confianza suficiente
        person_detections = []
        for i in range(len(classes)):
            if (classes[i] == PERSON_CLASS_ID and 
                confidence[i] >= self.confidence_threshold):
                
                bbox = bboxes[i].tolist()
                centroid = calculate_centroid(bbox)
                
                person_detections.append({
                    'bbox': bbox,
                    'centroid': centroid,
                    'confidence': confidence[i]
                })
        
        return person_detections