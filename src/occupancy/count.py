from datetime import datetime

class OccupancyCounter:
    def __init__(self, max_capacity=None):
        self.current_ids = set()
        self.max_capacity = max_capacity
        self.history = []
        self.alerts = []
        self.current_count = 0  # Mantenemos el conteo actual

    def update_ids(self, detected_ids):
        """
        Actualiza el conteo actual basada en un conjunto de IDs únicos detectados.
        detected_ids: lista o conjunto de identificadores únicos (por ejemplo, [1, 2, 3])
        """
        self.current_ids = set(detected_ids)
        self.current_count = len(self.current_ids)

        # Registrar el historial de conteo
        self.history.append({
            'timestamp': datetime.now(),
            'count': self.current_count
        })

        # Verificar si debemos generar alertas
        self._check_alerts()

    def _check_alerts(self):
        """Genera alertas si se excede la capacidad máxima"""
        if self.max_capacity and self.current_count > self.max_capacity:
            alert = {
                'timestamp': datetime.now(),
                'type': 'CAPACITY_EXCEEDED',
                'message': f'Capacidad excedida: {self.current_count}/{self.max_capacity}',
                'severity': 'HIGH'
            }
            self.alerts.append(alert)

    def get_current_count(self):
        """Retorna el conteo actual de personas únicas"""
        return self.current_count

    def get_alerts(self):
        """Retorna y limpia la lista de alertas pendientes"""
        alerts = self.alerts.copy()
        self.alerts.clear()
        return alerts

    def get_statistics(self):
        """Retorna estadísticas basadas en el historial de conteo"""
        if not self.history:
            return None
        counts = [record['count'] for record in self.history]
        return {
            'current': self.current_count,
            'max': max(counts),
            'min': min(counts),
            'average': sum(counts) / len(counts),
            'total_records': len(self.history)
        }