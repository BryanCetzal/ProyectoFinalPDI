import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def convert_to_grayscale(image):
        """RF2: Conversión a escala de grises."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def normalize_image(image):
        """RF2: Normalización de imagen."""
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    @staticmethod
    def remove_noise(image, kernel_size=5):
        """RF2: Eliminación de ruido usando filtro gaussiano."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def morphological_opening(image, kernel_size=5):
        """RF3: Operación morfológica - Apertura."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    @staticmethod
    def morphological_closing(image, kernel_size=5):
        """RF3: Operación morfológica - Cierre."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    @staticmethod
    def erosion(image, kernel_size=5):
        """RF3: Operación morfológica - Erosión."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(image, kernel, iterations=1)
    
    @staticmethod
    def dilation(image, kernel_size=5):
        """RF3: Operación morfológica - Dilatación."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)
    
    @staticmethod
    def histogram_equalization(image, use_adaptive=False, clip_limit=2.0, grid_size=8):
        """RF4: Ecualización de histograma con parámetros ajustables."""
        if use_adaptive:
            # Ecualización adaptativa (CLAHE)
            if len(image.shape) == 3:
                # Imagen en color
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Imagen en escala de grises
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
                return clahe.apply(image)
        else:
            # Ecualización normal
            if len(image.shape) == 3:
                # Imagen en color - convertir a YUV y ecualizar canal Y
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                # Imagen en escala de grises
                return cv2.equalizeHist(image)