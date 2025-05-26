import numpy as np
from scipy.optimize import linear_sum_assignment
from src.utils.helper import calculate_distance  # Si deseas seguir usando esta función en otros contextos.
from src.utils.constants import MAX_TRACKING_DISTANCE


class SimpleTracker:
    def __init__(self, max_frames_without_detection=10):
        """
        Inicializa el tracker simple.

        Args:
            max_frames_without_detection (int): Número máximo de frames sin detección
                                                  antes de eliminar un objeto.
        """
        self.tracked_objects = {}  # Diccionario de objetos actualmente rastreados
        self.next_id = 0  # ID siguiente a asignar
        self.max_frames_without_detection = max_frames_without_detection

    def update(self, detections):
        """
        Actualiza el tracking de objetos con las nuevas detecciones.

        Args:
            detections (list): Lista de diccionarios de detección, en cada elemento se espera
                               al menos {'centroid': (x, y), 'bbox': (startX, startY, endX, endY)}.

        Returns:
            int: Número de objetos actualmente rastreados.
        """
        # Si no hay detecciones, se actualizan los frames sin detección y se eliminan los perdidos.
        if len(detections) == 0:
            to_remove = []
            for obj_id, obj_data in self.tracked_objects.items():
                obj_data['frames_without_detection'] += 1
                if obj_data['frames_without_detection'] > self.max_frames_without_detection:
                    to_remove.append(obj_id)
            for obj_id in to_remove:
                del self.tracked_objects[obj_id]
            return len(self.tracked_objects)

        # Obtener los centroides de las detecciones actuales.
        detection_centroids = np.array([det['centroid'] for det in detections])

        # Si no existen objetos rastreados, se registran todas las detecciones.
        if len(self.tracked_objects) == 0:
            for detection in detections:
                self.tracked_objects[self.next_id] = {
                    'centroid': detection['centroid'],
                    'bbox': detection['bbox'],
                    'frames_without_detection': 0
                }
                self.next_id += 1
            return len(self.tracked_objects)

        # Preparar la lista de centroides de los objetos actuales.
        tracked_ids = list(self.tracked_objects.keys())
        tracked_centroids = np.array([self.tracked_objects[obj_id]['centroid'] for obj_id in tracked_ids])

        # Calcular la matriz de costos (distancias Euclidianas) entre los centroides ya registrados y
        # los centroides de las nuevas detecciones.
        D = np.linalg.norm(tracked_centroids[:, np.newaxis] - detection_centroids, axis=2)

        # Ejecutar la asignación óptima usando el algoritmo húngaro.
        rows, cols = linear_sum_assignment(D)

        # Llevar un registro de los objetos y detecciones que ya fueron asignados.
        assigned_tracked = set()
        assigned_detections = set()

        # Recorrer las asignaciones (cada par representa: fila=índice en tracked_centroids, col=índice en detecciones)
        for row, col in zip(rows, cols):
            # Solo se actualiza si la distancia es aceptable.
            if D[row, col] <= MAX_TRACKING_DISTANCE:
                obj_id = tracked_ids[row]
                self.tracked_objects[obj_id]['centroid'] = detections[col]['centroid']
                self.tracked_objects[obj_id]['bbox'] = detections[col]['bbox']
                self.tracked_objects[obj_id]['frames_without_detection'] = 0
                assigned_tracked.add(obj_id)
                assigned_detections.add(col)

        # Para los objetos no asignados, se incrementa su contador de frames sin detección.
        for obj_id in tracked_ids:
            if obj_id not in assigned_tracked:
                self.tracked_objects[obj_id]['frames_without_detection'] += 1

        # Eliminar los objetos que han estado demasiado tiempo sin ser detectados.
        remove_ids = [obj_id for obj_id, obj_data in self.tracked_objects.items()
                      if obj_data['frames_without_detection'] > self.max_frames_without_detection]
        for obj_id in remove_ids:
            del self.tracked_objects[obj_id]

        # Registrar nuevas detecciones que no hayan sido asignadas.
        for i, detection in enumerate(detections):
            if i not in assigned_detections:
                self.tracked_objects[self.next_id] = {
                    'centroid': detection['centroid'],
                    'bbox': detection['bbox'],
                    'frames_without_detection': 0
                }
                self.next_id += 1

        return len(self.tracked_objects)

    def get_tracked_objects(self):
        """
        Retorna los objetos actualmente rastreados con sus datos asociados.

        Returns:
            dict: Diccionario con la información de cada objeto rastreado.
        """
        return self.tracked_objects