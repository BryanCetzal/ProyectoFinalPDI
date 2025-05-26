from datetime import datetime
import json

class OccupancyCounter:
    def __init__(self, max_capacity=None):
        """
        Inicializa el contador de ocupación.

        Args:
            max_capacity (int): Capacidad máxima del espacio
        """
        self.current_ids = set()
        self.max_capacity = max_capacity
        self.history = []
        self.alerts = []
        self.current_count = 0
        self.peak_count = 0
        self.alert_thresholds = {
            'warning': 0.75,  # 75% de capacidad
            'critical': 0.90,  # 90% de capacidad
            'full': 1.0  # 100% de capacidad
        }
        self._last_alert_level = None

    def update_ids(self, detected_ids):
        """
        Actualiza el conteo actual basada en un conjunto de IDs únicos detectados.

        Args:
            detected_ids (list): Lista de identificadores únicos detectados
        """
        if isinstance(detected_ids, (list, tuple)):
            self.current_ids = set(detected_ids)
        else:
            self.current_ids = set([detected_ids]) if detected_ids is not None else set()

        self.current_count = len(self.current_ids)

        # Actualizar el pico máximo
        if self.current_count > self.peak_count:
            self.peak_count = self.current_count

        # Registrar el historial de conteo
        self.history.append({
            'timestamp': datetime.now(),
            'count': self.current_count,
            'ids': list(self.current_ids)
        })

        # Verificar alertas
        self._check_alerts()

    def set_max_capacity(self, capacity):
        """
        Establece la capacidad máxima del espacio.

        Args:
            capacity (int): Nueva capacidad máxima
        """
        if capacity <= 0:
            raise ValueError("La capacidad debe ser mayor a 0")

        self.max_capacity = capacity
        # Revaluar alertas con la nueva capacidad
        self._check_alerts()

    def get_remaining_capacity(self):
        """
        Retorna cuántos espacios quedan disponibles.

        Returns:
            int: Espacios disponibles, o None si no hay capacidad establecida
        """
        if self.max_capacity is None:
            return None
        return max(0, self.max_capacity - self.current_count)

    def get_occupancy_percentage(self):
        """
        Retorna el porcentaje de ocupación actual.

        Returns:
            float: Porcentaje de ocupación, o None si no hay capacidad establecida
        """
        if self.max_capacity is None or self.max_capacity == 0:
            return None
        return min(100.0, (self.current_count / self.max_capacity) * 100.0)

    def is_at_capacity(self):
        """
        Verifica si el espacio está a capacidad máxima.

        Returns:
            bool: True si está lleno, False en caso contrario
        """
        if self.max_capacity is None:
            return False
        return self.current_count >= self.max_capacity

    def is_over_capacity(self):
        """
        Verifica si el espacio está sobre la capacidad máxima.

        Returns:
            bool: True si está sobre capacidad, False en caso contrario
        """
        if self.max_capacity is None:
            return False
        return self.current_count > self.max_capacity

    def _check_alerts(self):
        """Genera alertas basadas en el nivel de ocupación actual"""
        if self.max_capacity is None or self.max_capacity == 0:
            return

        occupancy_percentage = self.get_occupancy_percentage()
        current_alert_level = None

        # Determinar el nivel de alerta actual
        if occupancy_percentage >= 100:
            current_alert_level = 'full'
        elif occupancy_percentage >= self.alert_thresholds['critical'] * 100:
            current_alert_level = 'critical'
        elif occupancy_percentage >= self.alert_thresholds['warning'] * 100:
            current_alert_level = 'warning'

        # Solo generar alerta si cambió el nivel
        if current_alert_level != self._last_alert_level and current_alert_level is not None:
            self._generate_alert(current_alert_level, occupancy_percentage)
            self._last_alert_level = current_alert_level
        elif current_alert_level is None:
            self._last_alert_level = None

    def _generate_alert(self, level, occupancy_percentage):
        """
        Genera una alerta específica basada en el nivel.

        Args:
            level (str): Nivel de alerta ('warning', 'critical', 'full')
            occupancy_percentage (float): Porcentaje actual de ocupación
        """
        alert_messages = {
            'warning': f'ADVERTENCIA: Alta ocupación detectada ({occupancy_percentage:.1f}%)',
            'critical': f'CRÍTICO: Capacidad casi completa ({occupancy_percentage:.1f}%)',
            'full': f'ALERTA MÁXIMA: Capacidad completa o excedida ({occupancy_percentage:.1f}%)'
        }

        alert_severities = {
            'warning': 'MEDIUM',
            'critical': 'HIGH',
            'full': 'CRITICAL'
        }

        alert = {
            'timestamp': datetime.now(),
            'type': f'OCCUPANCY_{level.upper()}',
            'message': alert_messages[level],
            'severity': alert_severities[level],
            'current_count': self.current_count,
            'max_capacity': self.max_capacity,
            'occupancy_percentage': occupancy_percentage,
            'remaining_capacity': self.get_remaining_capacity()
        }

        self.alerts.append(alert)

    def get_current_count(self):
        """Retorna el conteo actual de personas únicas"""
        return self.current_count

    def get_peak_count(self):
        """Retorna el conteo máximo registrado en esta sesión"""
        return self.peak_count

    def get_alerts(self):
        """Retorna y limpia la lista de alertas pendientes"""
        alerts = self.alerts.copy()
        self.alerts.clear()
        return alerts

    def get_active_ids(self):
        """Retorna los IDs actualmente detectados"""
        return list(self.current_ids)

    def get_statistics(self):
        """Retorna estadísticas basadas en el historial de conteo"""
        if not self.history:
            return None

        counts = [record['count'] for record in self.history]
        timestamps = [record['timestamp'] for record in self.history]

        # Calcular duración de la sesión
        session_duration = None
        if len(timestamps) > 1:
            session_duration = timestamps[-1] - timestamps[0]

        stats = {
            'current': self.current_count,
            'max': max(counts),
            'min': min(counts),
            'average': sum(counts) / len(counts),
            'peak_session': self.peak_count,
            'total_records': len(self.history),
            'max_capacity': self.max_capacity,
            'current_occupancy_percentage': self.get_occupancy_percentage(),
            'remaining_capacity': self.get_remaining_capacity(),
            'session_start': timestamps[0] if timestamps else None,
            'session_duration': session_duration,
            'is_at_capacity': self.is_at_capacity(),
            'is_over_capacity': self.is_over_capacity()
        }

        return stats

    def get_hourly_statistics(self):
        """Retorna estadísticas agrupadas por hora"""
        if not self.history:
            return None

        hourly_data = {}

        for record in self.history:
            hour_key = record['timestamp'].strftime('%Y-%m-%d %H:00')
            if hour_key not in hourly_data:
                hourly_data[hour_key] = []
            hourly_data[hour_key].append(record['count'])

        hourly_stats = {}
        for hour, counts in hourly_data.items():
            hourly_stats[hour] = {
                'average': sum(counts) / len(counts),
                'max': max(counts),
                'min': min(counts),
                'samples': len(counts)
            }

        return hourly_stats

    def export_data(self, filepath):
        """
        Exporta los datos históricos a un archivo JSON.

        Args:
            filepath (str): Ruta del archivo donde guardar los datos
        """
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'max_capacity': self.max_capacity,
            'current_count': self.current_count,
            'peak_count': self.peak_count,
            'statistics': self.get_statistics(),
            'history': []
        }

        # Convertir timestamps a strings para JSON
        for record in self.history:
            data['history'].append({
                'timestamp': record['timestamp'].isoformat(),
                'count': record['count'],
                'ids': record['ids']
            })

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exportando datos: {e}")
            return False

    def reset_session(self):
        """Reinicia los datos de la sesión actual"""
        self.current_ids = set()
        self.current_count = 0
        self.peak_count = 0
        self.history = []
        self.alerts = []
        self._last_alert_level = None

    def configure_alert_thresholds(self, warning=0.75, critical=0.90):
        """
        Configura los umbrales de alerta personalizados.

        Args:
            warning (float): Porcentaje para alerta de advertencia (0-1)
            critical (float): Porcentaje para alerta crítica (0-1)
        """
        if not (0 < warning < critical < 1):
            raise ValueError("Los umbrales deben estar entre 0 y 1, y warning < critical")

        self.alert_thresholds['warning'] = warning
        self.alert_thresholds['critical'] = critical

    def get_capacity_status(self):
        """
        Retorna un resumen del estado actual de capacidad.

        Returns:
            dict: Diccionario con información completa del estado
        """
        return {
            'current_count': self.current_count,
            'max_capacity': self.max_capacity,
            'remaining_capacity': self.get_remaining_capacity(),
            'occupancy_percentage': self.get_occupancy_percentage(),
            'is_at_capacity': self.is_at_capacity(),
            'is_over_capacity': self.is_over_capacity(),
            'peak_count': self.peak_count,
            'active_ids': self.get_active_ids(),
            'alert_level': self._last_alert_level
        }