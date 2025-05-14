from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import io
import numpy as np
import onnxruntime as ort
from PIL import Image
import logging
import os

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
        model = BirdClassifier(model_path)
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

# Clase para el clasificador
class BirdClassifier:
    def __init__(self, model_path):
        try:
            # Optimizar para servidor
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Inicializar la sesión
            self.session = ort.InferenceSession(model_path, session_options)
            
            # Obtener metadatos
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            # Cargar etiquetas
            self.labels_path = os.environ.get("LABELS_PATH", "model/bird_labels.txt")
            self.load_labels()
            
            logger.info(f"Modelo cargado. Forma de entrada: {self.input_shape}")
        except Exception as e:
            logger.error(f"Error en la inicialización: {str(e)}")
            raise e
    
    def load_labels(self):
        try:
            with open(self.labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        except Exception as e:
            logger.warning(f"No se pudieron cargar las etiquetas: {str(e)}")
            # Etiquetas de respaldo
            self.labels = ["especie_desconocida"]
    
    def preprocess_image(self, image):
        # Redimensionar
        target_size = (224, 224)
        if self.input_shape and len(self.input_shape) >= 3:
            if self.input_shape[2] > 0 and self.input_shape[3] > 0:
                target_size = (self.input_shape[2], self.input_shape[3])
        
        image = image.resize(target_size)
        
        # Convertir a RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Normalizar
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Reorganizar a NCHW
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Añadir dimensión de lote
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image):
        try:
            input_data = self.preprocess_image(image)
            inputs = {self.input_name: input_data}
            outputs = self.session.run([self.output_name], inputs)
            predictions = outputs[0][0]
            
            # Top 5 predicciones
            top_indices = np.argsort(predictions)[::-1][:5]
            
            results = []
            for idx in top_indices:
                if idx < len(self.labels):
                    results.append({
                        "species": self.labels[idx],
                        "confidence": float(predictions[idx])
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error durante la inferencia: {str(e)}")
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
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Verificar que el archivo es una imagen
        content_type = file.content_type
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Leer la imagen
        contents = await file.read()
        
        # Procesar la imagen
        image = Image.open(io.BytesIO(contents))
        
        # Realizar la predicción
        results = model.predict(image)
        
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
    
    status = "healthy" if model is not None else "model_not_loaded"
    
    return {
        "status": status,
        "model_info": {
            "loaded": model is not None,
            "input_shape": model.input_shape if model else None,
            "num_species": len(model.labels) if model else 0
        }
    }