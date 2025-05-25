import logging as py_logging
import os
from datetime import datetime

class SystemLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Configurar logger
        self.logger = py_logging.getLogger("PersonDetectionSystem")
        self.logger.setLevel(py_logging.INFO)
        
        # Limpiar handlers existentes
        self.logger.handlers.clear()
        
        # Handler para archivo
        log_file = os.path.join(log_dir, f"system_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = py_logging.FileHandler(log_file)
        file_handler.setLevel(py_logging.INFO)
        
        # Handler para consola
        console_handler = py_logging.StreamHandler()
        console_handler.setLevel(py_logging.INFO)
        
        # Formato
        formatter = py_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def warning(self, message):
        self.logger.warning(message)