"""
probar_roboflow.py
Script SUELTO, solo para probar la conexion con Roboflow directamente
y ver el error de verdad si algo falla -- deteccion_general_roboflow.py
esta disenado para fallar en silencio dentro de la app (para no
tumbarla), asi que desde main.py o app_web.py nunca ven el motivo
exacto si algo sale mal. Este script si muestra todo.

USO:
    python probar_roboflow.py postes_test/caso09.jpg
"""

import sys
import os
import cv2

WORKSPACE_NAME = "rodrigo-hoyos"            # debe coincidir con deteccion_general_roboflow.py
WORKFLOW_ID = "general-segmentation-api-2"  # debe coincidir con deteccion_general_roboflow.py


def main():
    if len(sys.argv) < 2:
        print("Uso: python probar_roboflow.py ruta/a/una/foto.jpg")
        sys.exit(1)

    ruta_imagen = sys.argv[1]
    if not os.path.exists(ruta_imagen):
        print(f"No encuentro el archivo: {ruta_imagen}")
        sys.exit(1)

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    print(f"1. API key encontrada: {'SI' if api_key else 'NO -- corre: set ROBOFLOW_API_KEY=tu-key'}")
    if not api_key:
        sys.exit(1)

    try:
        from inference_sdk import InferenceHTTPClient
        print("2. Libreria inference_sdk: OK")
    except ImportError:
        print("2. Libreria inference_sdk: NO INSTALADA -- corre: pip install inference-sdk")
        sys.exit(1)

    imagen = cv2.imread(ruta_imagen)
    print(f"3. Imagen cargada: {'OK' if imagen is not None else 'ERROR al abrir el archivo'}")

    print(f"4. Llamando a Roboflow (workspace={WORKSPACE_NAME}, workflow={WORKFLOW_ID})...")
    try:
        cliente = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
        resultado = cliente.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": ruta_imagen},
            parameters={"classes": "vehicle, human, Dori, Low-Hanging-Wire, tangled-wires"},
            use_cache=True,
        )
        print("5. Respuesta recibida SIN errores. Contenido completo:")
        print(resultado)
    except Exception as exc:
        print("5. ERROR al llamar a Roboflow:")
        print(f"   {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
