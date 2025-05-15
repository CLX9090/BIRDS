from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import io
import numpy as np
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
            self.net = cv2.dnn.readNetFromONNX("./model/bird_classifier.onnx")
            
            # Cargar etiquetas
            self.labels_path = os.environ.get("LABELS_PATH", "model/bird_labels.txt")
            self.load_labels()
        
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
            self.labels = []
    
    def preprocess_image(self, image):
        try:

            img = cv2.imdecode(np.array(image), -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(640,640))
            img = img.transpose(2,0,1)
            img = img.reshape(1,3,640,640)
            img = img/255.0
            return img
        except Exception as e:
            logger.error(f"Error en el preprocesamiento: {str(e)}")
            raise e
    def detect(self, image):
        try:
            self.net.setInput(image)
            out = self.net.forward()
            results = out[0]
            results = results.transpose()
            A = []
            for detection in results:
                class_id = detection[4:].argmax()
                confidence_score = detection[4:].max()
                new_detection = np.append(detection[:4],[class_id,confidence_score])
                A.append(new_detection)
                considerable_detections = [detection for detection in A if detection[-1] > 0.6]
                considerable_detections = np.array(considerable_detections)
            A = np.array(A)
            results = []
            species = []
            for detection in considerable_detections:
                if self.labels[int(detection[4])] not in species:
                    results.append({"species":self.labels[int(detection[4])], "confidence":round(float(detection[5]),3)})
                    species.append(self.labels[int(detection[4])])
                else:
                    pass
            if len(results) >= 1:
                return results
            else:
                logger.info("No se detectaron aves en la imagen")
                return "No se detectaron aves en la imagen"
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
        
        # Procesar la image
        contents = np.asarray(bytearray(contents), dtype="uint8")
        preprocess = model.preprocess_image(contents)
        results = model.detect(preprocess)
        print(results)
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
            "num_species": len(model.labels) if model else 0
        }
    }