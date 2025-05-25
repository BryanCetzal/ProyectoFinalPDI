import numpy as np
from src.utils.helper import calculate_distance
from src.utils.constants import MAX_TRACKING_DISTANCE

class SimpleTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0
        self.max_frames_without_detection = 10
    
    def update(self, detections):
        """Actualiza el tracking con nuevas detecciones."""
        current_centroids = [det['centroid'] for det in detections]
        
        # Actualizar objetos existentes
        updated_ids = set()
        for obj_id, obj_data in list(self.tracked_objects.items()):
            last_centroid = obj_data['centroid']
            
            # Buscar la detección más cercana
            min_distance = float('inf')
            closest_idx = -1
            
            for i, centroid in enumerate(current_centroids):
                distance = calculate_distance(last_centroid, centroid)
                if distance < min_distance and distance < MAX_TRACKING_DISTANCE:
                    min_distance = distance
                    closest_idx = i
            
            if closest_idx != -1:
                # Actualizar objeto existente
                self.tracked_objects[obj_id]['centroid'] = current_centroids[closest_idx]
                self.tracked_objects[obj_id]['bbox'] = detections[closest_idx]['bbox']
                self.tracked_objects[obj_id]['frames_without_detection'] = 0
                updated_ids.add(closest_idx)
            else:
                # Incrementar frames sin detección
                self.tracked_objects[obj_id]['frames_without_detection'] += 1
        
        # Eliminar objetos perdidos
        to_remove = []
        for obj_id, obj_data in self.tracked_objects.items():
            if obj_data['frames_without_detection'] > self.max_frames_without_detection:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
        
        # Agregar nuevas detecciones
        for i, detection in enumerate(detections):
            if i not in updated_ids:
                self.tracked_objects[self.next_id] = {
                    'centroid': detection['centroid'],
                    'bbox': detection['bbox'],
                    'frames_without_detection': 0
                }
                self.next_id += 1
        
        return len(self.tracked_objects)
    
    def get_tracked_objects(self):
        """Obtiene los objetos actualmente trackeados."""
        return self.tracked_objects
