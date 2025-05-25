from datetime import datetime

class OccupancyCounter:
    def __init__(self, max_capacity=None):
        self.current_count = 0
        self.max_capacity = max_capacity
        self.history = []
        self.alerts = []
    
    def update_count(self, detected_count):
        """Actualiza el conteo actual."""
        self.current_count = detected_count
        
        # Registrar en historial
        self.history.append({
            'timestamp': datetime.now(),
            'count': detected_count
        })
        
        # Verificar alertas
        self._check_alerts()
    
    def _check_alerts(self):
        """Verifica si se deben generar alertas."""
        if self.max_capacity and self.current_count > self.max_capacity:
            alert = {
                'timestamp': datetime.now(),
                'type': 'CAPACITY_EXCEEDED',
                'message': f'Capacidad excedida: {self.current_count}/{self.max_capacity}',
                'severity': 'HIGH'
            }
            self.alerts.append(alert)
    
    def get_current_count(self):
        """Obtiene el conteo actual."""
        return self.current_count
    
    def get_alerts(self):
        """Obtiene las alertas pendientes."""
        alerts = self.alerts.copy()
        self.alerts.clear()  # Limpiar alertas después de obtenerlas
        return alerts
    
    def get_statistics(self):
        """Obtiene estadísticas del conteo."""
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