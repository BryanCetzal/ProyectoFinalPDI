import cv2
import threading
import time

class CameraCapture:
    def __init__(self, source=0, width=1280, height=720):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def initialize(self):
        """Inicializa la cámara."""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise Exception(f"No se pudo abrir la cámara/video: {self.source}")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            return True
        except Exception as e:
            print(f"Error inicializando cámara: {e}")
            return False
    
    def start_capture(self):
        """Inicia la captura en un hilo separado."""
        if self.cap is None:
            if not self.initialize():
                return False
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def _capture_loop(self):
        """Bucle de captura de frames."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame.copy()
                self.frame_count += 1
                
                # Calcular FPS
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()
            else:
                time.sleep(0.01)
    
    def get_frame(self):
        """Obtiene el frame actual."""
        return self.frame.copy() if self.frame is not None else None
    
    def get_fps(self):
        """Obtiene los FPS actuales."""
        return self.fps
    
    def stop_capture(self):
        """Detiene la captura."""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()