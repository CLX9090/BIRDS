from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import io
import numpy as np
import onnxruntime as ort
from PIL import Image
import logging
import os
import cv2

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variable global para el modelo
model = None

# Gestor del ciclo de vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        # Cargar el modelo ONNX
        model_path = os.environ.get("MODEL_PATH", "model/bird_classifier.onnx")
        logger.info(f"Cargando modelo desde: {model_path}")
        model = BirdDetector(model_path)
        logger.info("Modelo cargado y listo para inferencia")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
    
    yield
    
    logger.info("Cerrando la aplicación")

# Inicializar FastAPI
app = FastAPI(
    title="Bird Species Detection API",
    description="API para detectar especies de aves a partir de imágenes",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clase para el detector de aves (YOLO)
class BirdDetector:
    def __init__(self, model_path):
        try:
            # Obtener proveedores disponibles
            available_providers = ort.get_available_providers()
            logger.info(f"Proveedores ONNX disponibles: {available_providers}")
            
            # Seleccionar proveedores adecuados
            providers = ['CPUExecutionProvider']
            
            # Optimizar para servidor
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Inicializar la sesión con proveedores explícitos
            self.session = ort.InferenceSession(
                model_path, 
                sess_options=session_options,
                providers=providers
            )
            
            # Obtener metadatos
            self.inputs = self.session.get_inputs()
            self.outputs = self.session.get_outputs()
            self.input_name = self.inputs[0].name
            
            # Determinar la forma de entrada
            self.input_shape = self.inputs[0].shape
            self.img_size = 640  # Tamaño por defecto para YOLO
            if len(self.input_shape) >= 4:
                if self.input_shape[2] > 0 and self.input_shape[3] > 0:
                    self.img_size = self.input_shape[2]  # Asumiendo que es cuadrado
            
            logger.info(f"Tamaño de imagen para el modelo: {self.img_size}x{self.img_size}")
            
            # Cargar etiquetas
            self.labels_path = os.environ.get("LABELS_PATH", "model/bird_labels.txt")
            self.load_labels()
            
            logger.info(f"Modelo cargado. Forma de entrada: {self.input_shape}")
            logger.info(f"Usando proveedores: {providers}")
        except Exception as e:
            logger.error(f"Error en la inicialización: {str(e)}")
            raise e
    
    def load_labels(self):
        try:
            with open(self.labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            logger.info(f"Etiquetas cargadas: {len(self.labels)} especies")
        except Exception as e:
            logger.warning(f"No se pudieron cargar las etiquetas: {str(e)}")
            # Etiquetas de respaldo
            self.labels = ["especie_desconocida"]
    
    def preprocess_image(self, image):
        try:
            # Convertir PIL Image a numpy array
            img = np.array(image)
            
            # Convertir a BGR (OpenCV format) si es RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Redimensionar y preparar para YOLO
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Normalizar y convertir a formato NCHW
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = np.expand_dims(img, axis=0)  # Añadir dimensión de lote
            
            return img
        except Exception as e:
            logger.error(f"Error en el preprocesamiento: {str(e)}")
            raise e
    
    def postprocess(self, outputs, conf_threshold=0.25, iou_threshold=0.45):
        try:
            # Extraer las detecciones
            predictions = outputs[0]  # Asumiendo que el primer output contiene las detecciones
            
            # Filtrar por confianza
            results = []
            
            # Procesar las predicciones según el formato de salida de YOLO
            # Este código puede necesitar ajustes según la estructura exacta de salida de tu modelo
            for i, pred in enumerate(predictions):
                if len(pred.shape) == 2:  # Formato [detecciones, (x, y, w, h, conf, cls1, cls2, ...)]
                    for detection in pred:
                        if len(detection) >= 6:  # Asegurarse de que hay suficientes elementos
                            confidence = detection[4]
                            if confidence >= conf_threshold:
                                class_id = int(detection[5])
                                if class_id < len(self.labels):
                                    results.append({
                                        "species": self.labels[class_id],
                                        "confidence": float(confidence)
                                    })
                elif len(pred.shape) == 3:  # Otro posible formato
                    # Implementar según sea necesario
                    pass
            
            # Ordenar por confianza
            results = sorted(results, key=lambda x: x["confidence"], reverse=True)
            
            # Limitar a las 5 mejores predicciones
            return results[:5]
        except Exception as e:
            logger.error(f"Error en el postprocesamiento: {str(e)}")
            raise e
    
    def detect(self, image):
        try:
            # Preprocesar la imagen
            input_data = self.preprocess_image(image)
            
            # Ejecutar inferencia
            logger.info("Ejecutando inferencia YOLO...")
            outputs = self.session.run(None, {self.input_name: input_data})
            
            # Postprocesar resultados
            results = self.postprocess(outputs)
            
            if results:
                logger.info(f"Detección completada. Top resultado: {results[0]['species']} ({results[0]['confidence']:.4f})")
            else:
                logger.info("No se detectaron aves en la imagen")
            
            return results
        except Exception as e:
            logger.error(f"Error durante la detección: {str(e)}")
            raise e

# ENDPOINT RAÍZ - Crucial para verificar que la API está funcionando
@app.get("/")
async def root():
    """Endpoint de bienvenida"""
    return {
        "message": "API de detección de especies de aves",
        "status": "online",
        "endpoints": {
            "/detect": "POST - Detectar especies de aves en una imagen",
            "/health": "GET - Verificar el estado de la API"
        }
    }

# Endpoint para detección
@app.post("/detect")
async def detect_bird(file: UploadFile = File(...)):
    """Detecta especies de aves en una imagen"""
    global model
    
    if model is None:
        logger.error("Solicitud de detección pero el modelo no está cargado")
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Verificar que el archivo es una imagen
        content_type = file.content_type
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        logger.info(f"Procesando imagen: {file.filename}, tipo: {content_type}")
        
        # Leer la imagen
        contents = await file.read()
        
        # Procesar la imagen
        image = Image.open(io.BytesIO(contents))
        
        # Realizar la detección
        results = model.detect(image)
        
        return {
            "predictions": results
        }
    except Exception as e:
        logger.error(f"Error en la detección: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento: {str(e)}")

# Endpoint de health check
@app.get("/health")
async def health_check():
    """Verificar el estado de la API"""
    global model
    
    status = "online" if model is not None else "model_not_loaded"
    
    return {
        "status": status,
        "model_info": {
            "loaded": model is not None,
            "input_shape": model.input_shape if model else None,
            "img_size": model.img_size if model else None,
            "num_species": len(model.labels) if model else 0
        }
    }