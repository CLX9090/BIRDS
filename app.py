import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from contextlib import asynccontextmanager

# Variable global para almacenar el modelo
model = None

# Definir el gestor de lifespan (reemplaza on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta al iniciar
    global model
    try:
        # Reemplaza 'model.onnx' con la ruta a tu modelo
        model_path = "modelo.onnx"
        model = ONNXModel(model_path)
        print("Modelo cargado y listo para inferencia")
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        # El servidor iniciará, pero el modelo no estará disponible
    
    yield  # Esto es donde la aplicación se ejecuta
    
    # Código que se ejecuta al cerrar (opcional)
    print("Cerrando la aplicación")

# Inicializar FastAPI con el gestor de lifespan
app = FastAPI(
    title="ONNX Model API", 
    description="API para inferencia de modelos ONNX",
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

# Clase para almacenar la sesión de ONNX Runtime
class ONNXModel:
    def __init__(self, model_path):
        """
        Inicializa la sesión de ONNX Runtime
        
        Args:
            model_path: Ruta al archivo del modelo ONNX
        """
        try:
            # Crear opciones de sesión para optimizar el rendimiento
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Inicializar la sesión de inferencia
            self.session = ort.InferenceSession(model_path, session_options)
            
            # Obtener metadatos del modelo
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Obtener información sobre la forma de entrada
            self.input_shape = self.session.get_inputs()[0].shape
            self.input_type = self.session.get_inputs()[0].type
            
            print(f"Modelo cargado con éxito. Forma de entrada: {self.input_shape}, Tipo: {self.input_type}")
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            raise e
    
    def predict(self, input_data):
        """
        Realiza una predicción con el modelo ONNX
        
        Args:
            input_data: Datos de entrada en formato numpy
            
        Returns:
            Resultado de la predicción
        """
        try:
            # Crear diccionario de entrada para la sesión
            inputs = {self.input_name: input_data}
            
            # Ejecutar la inferencia
            outputs = self.session.run([self.output_name], inputs)
            
            return outputs[0]
        except Exception as e:
            print(f"Error durante la inferencia: {str(e)}")
            raise e

# Modelo de datos para la entrada JSON
class PredictionInput(BaseModel):
    data: List[List[float]]
    shape: Optional[List[int]] = None

# Modelo de datos para la respuesta
class PredictionResponse(BaseModel):
    prediction: List
    shape: List[int]

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """
    Endpoint para realizar predicciones con datos JSON
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Convertir los datos de entrada a numpy array
        np_data = np.array(input_data.data, dtype=np.float32)
        
        # Reshape si se proporciona una forma específica
        if input_data.shape:
            np_data = np_data.reshape(input_data.shape)
        
        # Realizar la predicción
        result = model.predict(np_data)
        
        # Convertir el resultado a lista para la respuesta JSON
        prediction_list = result.tolist()
        
        return {
            "prediction": prediction_list,
            "shape": list(result.shape)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint para realizar predicciones con imágenes
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Leer la imagen
        contents = await file.read()
        
        # Procesar la imagen (ejemplo básico - ajustar según tu modelo)
        # Aquí deberías usar bibliotecas como PIL o OpenCV para el preprocesamiento real
        import numpy as np
        from PIL import Image
        
        image = Image.open(io.BytesIO(contents))
        # Redimensionar según los requisitos del modelo
        # Suponiendo que el modelo espera imágenes de 224x224
        image = image.resize((224, 224))
        # Convertir a array y normalizar
        img_array = np.array(image).astype(np.float32) / 255.0
        # Reorganizar a NCHW si es necesario (para modelos entrenados con frameworks como PyTorch)
        img_array = np.transpose(img_array, (2, 0, 1))
        # Añadir dimensión de lote
        img_array = np.expand_dims(img_array, axis=0)
        
        # Realizar la predicción
        result = model.predict(img_array)
        
        # Convertir el resultado a lista para la respuesta JSON
        prediction_list = result.tolist()
        
        return {
            "prediction": prediction_list,
            "shape": list(result.shape)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")

@app.get("/model/info")
async def model_info():
    """
    Endpoint para obtener información sobre el modelo cargado
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    return {
        "input_shape": model.input_shape,
        "input_type": model.input_type,
        "status": "loaded"
    }

if __name__ == "__main__":
    # Ejecutar el servidor con uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)