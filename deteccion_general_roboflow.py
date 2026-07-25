"""
deteccion_general_roboflow.py
Segunda opinion OPCIONAL usando el "Workflow" de Roboflow que detecta
varias clases en una sola llamada: Vehicle, Human, Dori,
Low-Hanging-Wire (cable colgando bajo) y Tangled-Wires (nudo de
cables). Reemplaza a deteccion_nudos_roboflow.py -- con este ya no
hace falta ese otro archivo, porque este cubre lo mismo (nudos) y ademas
persona y cable bajo, en una sola llamada a internet en vez de dos.

Mismo patron seguro que los demas modulos opcionales de este proyecto:
si falta la libreria, si no hay API key, o si algo falla -> devuelve
None, y pipeline.py sigue funcionando con lo que ya tenia (sin este
modulo, todo sigue igual que antes).

MUY IMPORTANTE ANTES DE USAR ESTO:
Si ya compartieron su API key de Roboflow en una captura de pantalla o
en un chat, esa clave hay que darla por expuesta. Vayan a Roboflow ->
Settings -> Roboflow API -> regeneren/roten la clave, y usen SOLO la
nueva a partir de ahora. Nunca escriban la clave directamente en este
archivo -- siempre como variable de entorno (ver Paso 1).

PASO 1 -- Guardar la API key (la nueva, despues de rotarla) como
variable de entorno:
    Windows (cmd):        set ROBOFLOW_API_KEY=tu-key-nueva-aqui
    Windows (PowerShell): $env:ROBOFLOW_API_KEY="tu-key-nueva-aqui"

PASO 2 -- Instalar la libreria (si ya la instalaron para el modulo
anterior, no hace falta repetir este paso):
    pip install inference-sdk

PASO 3 -- Confirmar workspace_name y workflow_id exactos: se ven en el
mismo bloque de codigo que les mostro Roboflow (donde dice
client.run_workflow(workspace_name=..., workflow_id=...)). Reemplacen
las dos constantes de aqui abajo por las suyas.
"""

import os
import tempfile

import cv2

WORKSPACE_NAME = "rodrigo-hoyos"            # <-- reemplazar por el suyo exacto
WORKFLOW_ID = "general-segmentation-api-2"  # <-- reemplazar por el suyo exacto


def _buscar_detecciones(dato, encontradas=None):
    """Los workflows de Roboflow pueden devolver la lista de
    detecciones anidada en distintos lugares segun como se armo el
    workflow. Esta funcion busca recursivamente cualquier diccionario
    que tenga 'class' y 'confidence', sin asumir una estructura fija
    -- asi no se rompe si el formato exacto de la respuesta varia."""
    if encontradas is None:
        encontradas = []

    if isinstance(dato, dict):
        if "class" in dato and "confidence" in dato:
            encontradas.append(dato)
        for valor in dato.values():
            _buscar_detecciones(valor, encontradas)
    elif isinstance(dato, list):
        for elemento in dato:
            _buscar_detecciones(elemento, encontradas)

    return encontradas


def analizar_con_workflow_general(imagen_bgr, confianza_minima=0.4):
    """Devuelve un dict con el conteo de cada clase detectada, o None
    si no se pudo usar por cualquier motivo. Ejemplo:
        {"human": 1, "tangled-wires": 3, "low-hanging-wire": 0,
         "vehicle": 0, "dori": 0}
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        return None

    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError:
        return None

    ruta_temporal = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as archivo_temp:
            ruta_temporal = archivo_temp.name
        cv2.imwrite(ruta_temporal, imagen_bgr)

        cliente = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
        resultado = cliente.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": ruta_temporal},
        )

        detecciones = _buscar_detecciones(resultado)
        conteo_por_clase = {}
        for deteccion in detecciones:
            if deteccion.get("confidence", 0) < confianza_minima:
                continue
            nombre_clase = str(deteccion.get("class", "desconocido")).lower()
            conteo_por_clase[nombre_clase] = conteo_por_clase.get(nombre_clase, 0) + 1

        return conteo_por_clase
    except Exception:
        return None
    finally:
        if ruta_temporal and os.path.exists(ruta_temporal):
            try:
                os.remove(ruta_temporal)
            except OSError:
                pass