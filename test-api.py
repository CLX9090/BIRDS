import requests
import sys
import os
from urllib.parse import urljoin

def test_api(api_url, image_path=None):
    """
    Prueba la API de detección de especies de aves
    
    Args:
        api_url (str): URL base de la API (ej: https://bird-species-api.onrender.com)
        image_path (str, optional): Ruta a la imagen para analizar
    """
    # Asegurarse de que la URL termina con /
    if not api_url.endswith('/'):
        api_url += '/'
    
    # Verificar que la API está funcionando
    try:
        print(f"Verificando estado de la API en {api_url}...")
        response = requests.get(urljoin(api_url, "health"), timeout=30)
        
        if response.status_code == 200:
            status_data = response.json()
            print(f"\n✅ Estado de la API: {status_data.get('status', 'desconocido')}")
            print(f"Mensaje: {status_data.get('message', '')}")
            
            # Mostrar endpoints disponibles
            if 'endpoints' in status_data:
                print("\nEndpoints disponibles:")
                for endpoint, desc in status_data['endpoints'].items():
                    print(f"  {endpoint}: {desc}")
        else:
            print(f"❌ Error al verificar el estado: Código {response.status_code}")
            print(response.text)
            return
    except Exception as e:
        print(f"❌ Error al conectar con la API: {e}")
        return
    
    # Si no se proporcionó una imagen, terminar aquí
    if not image_path:
        print("\n⚠️ No se proporcionó una imagen para analizar.")
        print("Uso: python test_api.py URL_API RUTA_IMAGEN")
        return
    
    # Verificar que el archivo existe
    if not os.path.exists(image_path):
        print(f"\n❌ Error: El archivo '{image_path}' no existe.")
        return
    
    # Enviar la imagen para detección
    try:
        print(f"\nEnviando imagen '{image_path}' para análisis...")
        
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(
                urljoin(api_url, "detect"), 
                files=files,
                timeout=60  # Aumentar timeout para dar tiempo al procesamiento
            )
        
        if response.status_code == 200:
            result = response.json()
            
            if 'predictions' in result and result['predictions']:
                print("\n🔍 Especies detectadas:")
                print("-" * 50)
                print(f"{'#':<4}{'Especie':<40}{'Confianza':<10}")
                print("-" * 50)
                
                for i, pred in enumerate(result['predictions'], 1):
                    confidence = pred['confidence'] * 100
                    print(f"{i:<4}{pred['species']:<40}{confidence:.2f}%")
            else:
                print("\n⚠️ No se detectaron especies de aves en esta imagen.")
        else:
            print(f"\n❌ Error en la detección: Código {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"\n❌ Error al procesar la imagen: {e}")

if __name__ == "__main__":
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("Uso: python test_api.py URL_API [RUTA_IMAGEN]")
        sys.exit(1)
    
    api_url = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_api(api_url, image_path)