import tkinter as tk
import sys
import os

# Agregar el directorio raíz al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """Función principal del sistema."""
    try:
        # Importar después de configurar el path
        from src.ui.interface import PersonDetectionGUI
        
        # Crear ventana principal
        root = tk.Tk()
        
        # Crear aplicación
        app = PersonDetectionGUI(root)
        
        # Configurar cierre de aplicación
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Iniciar bucle principal
        root.mainloop()
        
    except Exception as e:
        print(f"Error iniciando aplicación: {e}")
        import traceback
        traceback.print_exc()
        input("Presiona Enter para cerrar...")
        sys.exit(1)

if __name__ == "__main__":
    main()
