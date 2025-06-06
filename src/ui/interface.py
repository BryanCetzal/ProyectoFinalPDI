import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import time
import os


class PersonDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección y Conteo de Personas")
        self.root.geometry("1400x950")

        # Variables
        self.camera = None
        self.detector = None
        self.tracker = None
        self.counter = None
        self.processor = None
        self.logger = None

        self.running = False
        self.current_frame = None
        self.processed_frame = None

        # Variables de capacidad
        self.max_capacity = 0
        self.current_count = 0
        self.capacity_percentage = 0

        self.setup_ui()

    def setup_ui(self):
        """Configura la interfaz de usuario."""
        # Frame principal
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel de configuración de capacidad
        self.capacity_frame = ttk.LabelFrame(self.main_frame, text="Configuración de Capacidad")
        self.capacity_frame.pack(fill=tk.X, pady=(0, 10))

        # Controles de capacidad
        capacity_controls = ttk.Frame(self.capacity_frame)
        capacity_controls.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(capacity_controls, text="Capacidad máxima:").pack(side=tk.LEFT)

        self.capacity_var = tk.StringVar(value="0")
        self.capacity_entry = ttk.Entry(capacity_controls, textvariable=self.capacity_var, width=10)
        self.capacity_entry.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Button(capacity_controls, text="Establecer Capacidad",
                   command=self.set_capacity).pack(side=tk.LEFT, padx=(0, 10))

        # Información de capacidad
        self.capacity_info = ttk.Frame(self.capacity_frame)
        self.capacity_info.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.capacity_label = ttk.Label(self.capacity_info, text="Capacidad: No establecida",
                                        font=("Arial", 10, "bold"))
        self.capacity_label.pack(side=tk.LEFT)

        self.remaining_label = ttk.Label(self.capacity_info, text="Espacios libres: --")
        self.remaining_label.pack(side=tk.LEFT, padx=(20, 0))

        self.percentage_label = ttk.Label(self.capacity_info, text="Ocupación: 0%")
        self.percentage_label.pack(side=tk.LEFT, padx=(20, 0))

        # Barra de progreso
        progress_frame = ttk.Frame(self.capacity_frame)
        progress_frame.pack(fill=tk.X, padx=10, pady=(0, 5))

        ttk.Label(progress_frame, text="Nivel de ocupación:").pack(side=tk.LEFT)
        self.capacity_progress = ttk.Progressbar(progress_frame, mode='determinate', length=200)
        self.capacity_progress.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)

        # Panel de alertas
        self.alert_frame = ttk.Frame(self.capacity_frame)
        self.alert_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        self.alert_label = ttk.Label(self.alert_frame, text="",
                                     font=("Arial", 10, "bold"), foreground="green")
        self.alert_label.pack()

        # Panel de control superior
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=(0, 10))

        # Botones de control
        ttk.Button(self.control_frame, text="Iniciar Cámara",
                   command=self.start_camera).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.control_frame, text="Detener",
                   command=self.stop_camera).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.control_frame, text="Capturar Frame",
                   command=self.capture_frame).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.control_frame, text="Cargar Imagen",
                   command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.control_frame, text="Ver Estadísticas",
                   command=self.show_statistics).pack(side=tk.LEFT, padx=(0, 5))

        # Panel de video y procesamiento
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        # Video original
        self.original_label_frame = ttk.LabelFrame(self.video_frame, text="Video Original")
        self.original_label_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.original_label = ttk.Label(self.original_label_frame)
        self.original_label.pack(expand=True)

        # Panel de procesamiento
        self.processing_frame = ttk.LabelFrame(self.video_frame, text="Procesamiento de Imagen")
        self.processing_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Controles de procesamiento
        self.processing_controls = ttk.Frame(self.processing_frame)
        self.processing_controls.pack(fill=tk.X, padx=5, pady=5)

        # Filtros
        ttk.Button(self.processing_controls, text="Escala de Grises",
                   command=lambda: self.apply_filter('grayscale')).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.processing_controls, text="Eliminar Ruido",
                   command=lambda: self.apply_filter('denoise')).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.processing_controls, text="Ecualizar Histograma",
                   command=lambda: self.apply_filter('histogram')).pack(side=tk.LEFT, padx=2)

        # Segunda fila de filtros morfológicos
        self.morph_controls = ttk.Frame(self.processing_frame)
        self.morph_controls.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(self.morph_controls, text="Erosión",
                   command=lambda: self.apply_filter('erosion')).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.morph_controls, text="Dilatación",
                   command=lambda: self.apply_filter('dilation')).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.morph_controls, text="Apertura",
                   command=lambda: self.apply_filter('opening')).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.morph_controls, text="Cierre",
                   command=lambda: self.apply_filter('closing')).pack(side=tk.LEFT, padx=2)

        # Botón guardar
        ttk.Button(self.processing_controls, text="Guardar Imagen",
                   command=self.save_processed_image).pack(side=tk.RIGHT, padx=2)

        # Imagen procesada
        self.processed_label = ttk.Label(self.processing_frame)
        self.processed_label.pack(expand=True, fill=tk.BOTH)

        # Panel de información inferior
        self.info_frame = ttk.Frame(self.main_frame)
        self.info_frame.pack(fill=tk.X, pady=(10, 0))

        # Labels de información
        self.count_label = ttk.Label(self.info_frame, text="Personas detectadas: 0",
                                     font=("Arial", 12, "bold"))
        self.count_label.pack(side=tk.LEFT)

        self.fps_label = ttk.Label(self.info_frame, text="FPS: 0.0")
        self.fps_label.pack(side=tk.LEFT, padx=(20, 0))

        self.status_label = ttk.Label(self.info_frame, text="Estado: Detenido")
        self.status_label.pack(side=tk.RIGHT)

    def set_capacity(self):
        """Establece la capacidad máxima del espacio."""
        try:
            capacity = int(self.capacity_var.get())
            if capacity <= 0:
                messagebox.showerror("Error", "La capacidad debe ser un número mayor a 0")
                return

            self.max_capacity = capacity

            # Actualizar el contador con la nueva capacidad
            if self.counter:
                self.counter.max_capacity = capacity

            self.update_capacity_display()
            messagebox.showinfo("Éxito", f"Capacidad establecida: {capacity} personas")

            if self.logger:
                self.logger.info(f"Capacidad establecida: {capacity}")

        except ValueError:
            messagebox.showerror("Error", "Por favor ingrese un número válido")

    def update_capacity_display(self):
        """Actualiza la información de capacidad en la interfaz."""
        if self.max_capacity > 0:
            remaining = max(0, self.max_capacity - self.current_count)
            self.capacity_percentage = min(100, (self.current_count / self.max_capacity) * 100)

            # Actualizar labels
            self.capacity_label.config(text=f"Capacidad: {self.current_count}/{self.max_capacity}")
            self.remaining_label.config(text=f"Espacios libres: {remaining}")
            self.percentage_label.config(text=f"Ocupación: {self.capacity_percentage:.1f}%")

            # Actualizar barra de progreso
            self.capacity_progress['value'] = self.capacity_percentage

            # Actualizar alertas
            self.update_alerts()
        else:
            self.capacity_label.config(text="Capacidad: No establecida")
            self.remaining_label.config(text="Espacios libres: --")
            self.percentage_label.config(text="Ocupación: --")
            self.capacity_progress['value'] = 0
            self.alert_label.config(text="", foreground="green")

    def update_alerts(self):
        """Actualiza las alertas basadas en la ocupación."""
        if self.max_capacity <= 0:
            return

        if self.current_count >= self.max_capacity:
            # Capacidad completa
            self.alert_label.config(
                text="⚠️ CAPACIDAD COMPLETA - No se pueden admitir más personas",
                foreground="red"
            )
            # Reproducir sonido de alerta si está disponible
            self.root.bell()
        elif self.capacity_percentage >= 90:
            # Casi lleno
            self.alert_label.config(
                text="⚠️ ALERTA: Capacidad casi completa (>90%)",
                foreground="orange"
            )
        elif self.capacity_percentage >= 75:
            # Advertencia
            self.alert_label.config(
                text="⚠️ ADVERTENCIA: Alta ocupación (>75%)",
                foreground="#FF8C00"
            )
        else:
            # Normal
            self.alert_label.config(
                text="✅ Nivel de ocupación normal",
                foreground="green"
            )

    def show_statistics(self):
        """Muestra estadísticas detalladas en una ventana emergente."""
        if not self.counter:
            messagebox.showwarning("Advertencia", "No hay datos de estadísticas disponibles")
            return

        stats = self.counter.get_statistics()
        if not stats:
            messagebox.showinfo("Información", "No hay suficientes datos para mostrar estadísticas")
            return

        # Crear ventana de estadísticas
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Estadísticas de Ocupación")
        stats_window.geometry("400x300")
        stats_window.resizable(False, False)

        # Contenido de estadísticas
        ttk.Label(stats_window, text="📊 Estadísticas de Ocupación",
                  font=("Arial", 14, "bold")).pack(pady=10)

        stats_frame = ttk.Frame(stats_window)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Información actual
        ttk.Label(stats_frame, text="Información Actual:",
                  font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Label(stats_frame, text=f"• Personas actuales: {stats['current']}").pack(anchor="w")
        ttk.Label(stats_frame, text=f"• Capacidad máxima: {self.max_capacity}").pack(anchor="w")
        ttk.Label(stats_frame, text=f"• Ocupación actual: {self.capacity_percentage:.1f}%").pack(anchor="w")

        # Estadísticas históricas
        ttk.Separator(stats_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(stats_frame, text="Estadísticas Históricas:",
                  font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Label(stats_frame, text=f"• Máximo registrado: {stats['max']} personas").pack(anchor="w")
        ttk.Label(stats_frame, text=f"• Mínimo registrado: {stats['min']} personas").pack(anchor="w")
        ttk.Label(stats_frame, text=f"• Promedio: {stats['average']:.1f} personas").pack(anchor="w")
        ttk.Label(stats_frame, text=f"• Total de registros: {stats['total_records']}").pack(anchor="w")

        # Botón cerrar
        ttk.Button(stats_window, text="Cerrar",
                   command=stats_window.destroy).pack(pady=20)

    def initialize_components(self):
        """Inicializa los componentes del sistema."""
        try:
            from src.capture.camera import CameraCapture
            from src.detection.model_loader import ModelLoader
            from src.detection.detect import PersonDetector
            from src.tracking.object_tracker import SimpleTracker
            from src.occupancy.count import OccupancyCounter
            from src.preprocessing.image_preprocessing import ImageProcessor
            from src.system_logger import SystemLogger

            # Inicializar componentes
            self.camera = CameraCapture()

            model_loader = ModelLoader()
            if model_loader.load_model():
                self.detector = PersonDetector(model_loader.get_model())
                print("Modelo YOLO cargado exitosamente")
            else:
                print("No se pudo cargar el modelo YOLO - continuando sin detección")

            self.tracker = SimpleTracker()
            self.counter = OccupancyCounter(max_capacity=self.max_capacity)
            self.processor = ImageProcessor()
            self.logger = SystemLogger()

            return True
        except Exception as e:
            print(f"Error inicializando componentes: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start_camera(self):
        """Inicia la cámara y detección."""
        if not self.running:
            if not hasattr(self, 'camera') or self.camera is None:
                if not self.initialize_components():
                    messagebox.showerror("Error", "No se pudieron inicializar los componentes")
                    return

            if self.camera.start_capture():
                self.running = True
                self.status_label.config(text="Estado: Ejecutándose")
                self.update_video()
                if self.logger:
                    self.logger.info("Sistema iniciado")
            else:
                messagebox.showerror("Error", "No se pudo iniciar la cámara")

    def stop_camera(self):
        """Detiene la cámara y detección."""
        self.running = False
        if self.camera:
            self.camera.stop_capture()
        self.status_label.config(text="Estado: Detenido")
        if self.logger:
            self.logger.info("Sistema detenido")

    def update_video(self):
        """Actualiza el video en tiempo real."""
        if self.running and self.camera:
            frame = self.camera.get_frame()
            if frame is not None:
                self.current_frame = frame.copy()

                # Realizar detección si el detector está disponible
                if self.detector:
                    try:
                        detections = self.detector.detect_persons(frame)
                        count = self.tracker.update(detections)

                        # Obtener IDs actuales para el contador
                        tracked_objects = self.tracker.get_tracked_objects()
                        current_ids = list(tracked_objects.keys())

                        self.counter.update_ids(current_ids)
                        self.current_count = self.counter.get_current_count()

                        # Dibujar detecciones
                        self.draw_detections(frame, tracked_objects)

                        # Actualizar información
                        self.count_label.config(text=f"Personas detectadas: {self.current_count}")
                        self.update_capacity_display()

                        # Procesar alertas
                        alerts = self.counter.get_alerts()
                        for alert in alerts:
                            if self.logger:
                                self.logger.warning(alert['message'])

                    except Exception as e:
                        print(f"Error en detección: {e}")

                # Actualizar FPS
                self.fps_label.config(text=f"FPS: {self.camera.get_fps():.1f}")

                # Mostrar frame
                self.display_frame(frame, self.original_label)

        if self.running:
            self.root.after(50, self.update_video)

    def draw_detections(self, frame, tracked_objects):
        """Dibuja las detecciones en el frame."""
        COLORS = {
            'BOUNDING_BOX': (0, 0, 255),  # Rojo
            'CENTROID': (0, 255, 0),  # Verde
            'TEXT': (255, 255, 255),  # Blanco
        }

        for obj_id, obj_data in tracked_objects.items():
            bbox = obj_data['bbox']
            centroid = obj_data['centroid']

            # Dibujar bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['BOUNDING_BOX'], 2)

            # Dibujar centroide
            cv2.circle(frame, centroid, 5, COLORS['CENTROID'], -1)

            # Dibujar ID
            cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['TEXT'], 1)

        # Panel de información expandido
        panel_height = 100
        cv2.rectangle(frame, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, panel_height), (255, 255, 255), 2)

        # Información básica
        cv2.putText(frame, f'Personas: {len(tracked_objects)}', (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'FPS: {self.camera.get_fps():.1f}', (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Información de capacidad
        if self.max_capacity > 0:
            cv2.putText(frame, f'Capacidad: {self.current_count}/{self.max_capacity}', (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'Ocupacion: {self.capacity_percentage:.1f}%', (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def display_frame(self, frame, label_widget):
        """Muestra un frame en el widget especificado."""
        try:
            # Redimensionar frame para ajustarse al widget
            height, width = frame.shape[:2]
            max_width, max_height = 600, 400

            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # Convertir a formato compatible con Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)

            label_widget.config(image=photo)
            label_widget.image = photo
        except Exception as e:
            print(f"Error mostrando frame: {e}")

    def capture_frame(self):
        """Captura el frame actual."""
        if self.current_frame is not None:
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"frame_{timestamp}.png"
                os.makedirs("output", exist_ok=True)
                filepath = f"output/{filename}"
                cv2.imwrite(filepath, self.current_frame)
                messagebox.showinfo("Captura", f"Frame guardado en: {filepath}")
                if self.logger:
                    self.logger.info(f"Frame capturado: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Error guardando frame: {str(e)}")
        else:
            messagebox.showwarning("Advertencia", "No hay frame para capturar")

    def load_image(self):
        """Carga una imagen desde archivo."""
        filetypes = [
            ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("PNG", "*.png"),
            ("Todos los archivos", "*.*")
        ]

        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=filetypes
        )

        if filepath:
            try:
                image = cv2.imread(filepath)
                if image is not None:
                    self.current_frame = image
                    self.display_frame(image, self.original_label)
                    messagebox.showinfo("Éxito", "Imagen cargada correctamente")
                    if self.logger:
                        self.logger.info(f"Imagen cargada: {filepath}")
                else:
                    messagebox.showerror("Error", "No se pudo cargar la imagen")
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando imagen: {str(e)}")
                if self.logger:
                    self.logger.error(f"Error cargando imagen: {str(e)}")

    def apply_filter(self, filter_type):
        """Aplica filtros de procesamiento a la imagen actual."""
        if self.current_frame is None:
            messagebox.showwarning("Advertencia", "No hay imagen para procesar")
            return

        if not hasattr(self, 'processor') or self.processor is None:
            self.initialize_components()

        try:
            image = self.current_frame.copy()

            if filter_type == 'grayscale':
                processed = self.processor.convert_to_grayscale(image)
                # Convertir de vuelta a BGR para mostrar
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            elif filter_type == 'denoise':
                processed = self.processor.remove_noise(image)
            elif filter_type == 'histogram':
                processed = self.processor.histogram_equalization(image)
            elif filter_type == 'erosion':
                gray = self.processor.convert_to_grayscale(image)
                processed = self.processor.erosion(gray)
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            elif filter_type == 'dilation':
                gray = self.processor.convert_to_grayscale(image)
                processed = self.processor.dilation(gray)
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            elif filter_type == 'opening':
                gray = self.processor.convert_to_grayscale(image)
                processed = self.processor.morphological_opening(gray)
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            elif filter_type == 'closing':
                gray = self.processor.convert_to_grayscale(image)
                processed = self.processor.morphological_closing(gray)
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            else:
                processed = image

            self.processed_frame = processed
            self.display_frame(processed, self.processed_label)
            if self.logger:
                self.logger.info(f"Filtro aplicado: {filter_type}")

        except Exception as e:
            messagebox.showerror("Error", f"Error aplicando filtro: {str(e)}")
            if self.logger:
                self.logger.error(f"Error aplicando filtro {filter_type}: {str(e)}")

    def save_processed_image(self):
        """Guarda la imagen procesada."""
        if self.processed_frame is None:
            messagebox.showwarning("Advertencia", "No hay imagen procesada para guardar")
            return

        filetypes = [
            ("PNG", "*.png"),
            ("JPG", "*.jpg"),
            ("Todos los archivos", "*.*")
        ]

        filepath = filedialog.asksaveasfilename(
            title="Guardar imagen procesada",
            defaultextension=".png",
            filetypes=filetypes
        )

        if filepath:
            try:
                cv2.imwrite(filepath, self.processed_frame)
                messagebox.showinfo("Éxito", f"Imagen guardada en: {filepath}")
                if self.logger:
                    self.logger.info(f"Imagen procesada guardada: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Error guardando imagen: {str(e)}")
                if self.logger:
                    self.logger.error(f"Error guardando imagen: {str(e)}")

    def on_closing(self):
        """Maneja el cierre de la aplicación."""
        self.stop_camera()
        self.root.destroy()