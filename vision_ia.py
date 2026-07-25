import os

import cv2
import numpy as np
from ultralytics import YOLO

RUTA_BASE = os.path.dirname(os.path.abspath(__file__))

CANDIDATOS_MODELO = [
    ("YOLO26n", os.path.join(RUTA_BASE, "yolo26n.pt")),
    ("YOLOv8n", os.path.join(RUTA_BASE, "yolov8n.pt")),
]

model = None
NOMBRE_MODELO_ACTIVO = None
ERROR_CARGA_MODELO = None

for nombre, ruta_pesos in CANDIDATOS_MODELO:
    try:
        model = YOLO(ruta_pesos)
        NOMBRE_MODELO_ACTIVO = nombre
        break
    except Exception as exc:  # noqa: BLE001
        ERROR_CARGA_MODELO = str(exc)
        continue

if model is None:
    raise RuntimeError(
        "No se pudo cargar ningun modelo YOLO (se probo YOLO26n y YOLOv8n). "
        f"Ultimo error: {ERROR_CARGA_MODELO}. "
        "Verifica que 'yolo26n.pt' o 'yolov8n.pt' esten en la carpeta del "
        "proyecto, o que haya conexion a internet para descargarlos."
    )

CLASES_EVITAR = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
    10,  # fire hydrant
    11,  # stop sign
    13,  # bench
    56,  # chair
    58,  # potted plant
]

CONF_POR_DEFECTO = 0.28

# NUEVO: umbral de confianza especial y mas bajo para la clase
# "persona" (id 0 en COCO). Esto es una decision de seguridad, no de
# precision: preferimos una alerta de mas (un falso positivo marcando
# a alguien que en realidad no estaba tan cerca) que una persona real
# que no se detecta por estar parcialmente tapada entre cables --como
# el tecnico subido al poste en el banco de imagenes-- y que
# simplemente nunca dispare la alerta de "riesgo de contacto".
UMBRAL_MINIMO_PERSONA = 0.15


def analizar_escena(imagen, conf=CONF_POR_DEFECTO):
    """Devuelve una mascara limpia (sin ruido urbano ni vegetacion)
    y las cajas detectadas, para dibujarlas despues en la interfaz.

    BUG CORREGIDO: antes se corria YOLO con UN SOLO umbral de
    confianza para todas las clases (el mismo control "Sensibilidad
    IA" que ajusta el usuario). Eso tenia un efecto secundario
    peligroso: con la sensibilidad en su valor normal, una persona
    parcialmente oculta entre un amasijo de cables (poca luz, cuerpo
    tapado en gran parte) podia quedar justo por debajo del umbral y
    JAMAS aparecer en bboxes_yolo -- sin persona detectada, la alerta
    de "riesgo de contacto" (detectar_riesgo_contacto_persona en
    analisis.py) no tenia forma de dispararse, sin importar que tan
    bien programada estuviera esa logica.

    Ahora las personas se buscan con su propio umbral, mas bajo y fijo
    (no depende del control de sensibilidad general), mientras que el
    resto de clases de ruido (autos, semaforos, etc.) se sigue
    filtrando con el umbral que el usuario ajusta.
    """
    alto, ancho = imagen.shape[:2]
    mascara = np.ones((alto, ancho), dtype="uint8") * 255
    detecciones_yolo = []
    bboxes_vegetacion = []

    umbral_inferencia = min(conf, UMBRAL_MINIMO_PERSONA)
    try:
        resultados = model(imagen, conf=umbral_inferencia, verbose=False)
    except Exception:
        resultados = []

    for r in resultados:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in CLASES_EVITAR:
                continue

            confianza = float(box.conf[0])
            umbral_de_esta_clase = UMBRAL_MINIMO_PERSONA if cls_id == 0 else conf
            if confianza < umbral_de_esta_clase:
                continue

            nombre = model.names[cls_id].upper()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detecciones_yolo.append((x1, y1, x2, y2, nombre))
            cv2.rectangle(mascara, (x1, y1), (x2, y2), 0, -1)

    # BUG CORREGIDO: el indice "exceso de verde" (ExG = 2G - R - B) que
    # se usaba para detectar vegetacion tambien da un valor ALTO para
    # el color AMARILLO puro (mucho verde + mucho rojo, poco azul) --
    # matematicamente el amarillo "engaña" a la formula igual que el
    # verde real. Por eso una escalera amarilla de trabajo (como la del
    # tecnico en el banco de imagenes) se marcaba como "VEGETACION" en
    # vez de ignorarse sin mas. Ahora se exige ADEMAS que el tono (hue)
    # este realmente en el rango verde -- el amarillo tiene un hue mas
    # bajo y queda descartado, aunque su ExG sea alto.
    b, g, r_canal = cv2.split(imagen.astype(np.float32))
    exg = np.clip(2 * g - r_canal - b, 0, 255).astype(np.uint8)
    _, mask_exg = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    tono = hsv[:, :, 0]
    # En OpenCV el hue va de 0 a 179. El verde real cae entre ~35 y
    # ~95; el amarillo (lo que causaba el bug) esta por debajo de eso.
    mask_tono_verde = cv2.inRange(tono, 35, 95)

    mask_veg = cv2.bitwise_and(mask_exg, mask_tono_verde)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_veg = cv2.morphologyEx(mask_veg, cv2.MORPH_OPEN, kernel)
    mask_veg = cv2.morphologyEx(mask_veg, cv2.MORPH_DILATE, kernel, iterations=2)

    contornos_veg, _ = cv2.findContours(mask_veg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contornos_veg:
        if cv2.contourArea(cnt) > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes_vegetacion.append((x, y, x + w, y + h))
            cv2.drawContours(mascara, [cnt], -1, 0, -1)

    return mascara, detecciones_yolo, bboxes_vegetacion
