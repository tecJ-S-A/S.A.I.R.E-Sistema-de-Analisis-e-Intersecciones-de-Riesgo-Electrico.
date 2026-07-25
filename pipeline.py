"""
pipeline.py
Pipeline UNICO de analisis de S.A.I.R.E. PRO.

CAMBIO grande respecto a versiones anteriores: ya no se calcula "un
numero de cables". Se calcula un RANGO estimado (cables_min,
cables_max), la categoria de riesgo, el tipo de poste con los rangos
reales investigados, una recomendacion en texto y si hay senal de
"nudo" (posible conexion no regulada). Todo eso vive ahora en cada
diccionario de la lista "postes" que devuelve ejecutar_pipeline().
"""

import cv2

from vision_ia import analizar_escena, CONF_POR_DEFECTO
from procesamiento import preparar_y_obtener_bordes
from analisis import (
    detectar_estructura,
    detectar_cables_detallado,
    estimar_rango_cables,
    sanear_rango_cables,
    calcular_densidad_cables,
    clasificar_densidad,
    clasificar_tipo_poste,
    evaluar_riesgo,
    generar_mensaje_poste,
    detectar_riesgo_contacto_persona,
    contar_nudos_cables,
)
from modelo_ml import predecir_estimado_ml
from deteccion_general_roboflow import analizar_con_workflow_general

COLOR_CABLES = (0, 255, 0)
COLOR_POSTE = (255, 255, 0)
COLOR_BASURA = (0, 0, 255)
COLOR_VEGETACION = (0, 165, 255)


def ejecutar_pipeline(img_bruta, conf=CONF_POR_DEFECTO):
    """Devuelve:
        {
            "img_original":  imagen 1280x720 sin marcas,
            "img_procesada": imagen con detecciones dibujadas,
            "postes": [
                {
                    "tipo": "MEDIA TENSION" | "BAJA TENSION" | "USO MIXTO" | "REVISION MANUAL",
                    "cables_min": int,
                    "cables_max": int,
                    "cables_estimado": int,   # valor central, por si se necesita un solo numero
                    "estado": str,
                    "color_bgr": (b,g,r),
                    "recomendacion": str,
                    "nudo_detectado": bool,
                    "densidad_categoria": str,
                    "mensaje": str,            # texto listo para UI o PDF
                    "ancho_px": int,
                    "bbox": (x1,y1,x2,y2) | None,
                },
                ...
            ],
        }
    """
    img = cv2.resize(img_bruta, (1280, 720))
    img_original = img.copy()

    mascara, bboxes_yolo, bboxes_veg = analizar_escena(img, conf=conf)
    bordes = preparar_y_obtener_bordes(img, mascara)
    ancho_poste, bbox_poste = detectar_estructura(bordes)

    num_cables, contornos_cables, picos_izq, picos_der = detectar_cables_detallado(bordes, bbox_poste)

    densidad = calcular_densidad_cables(bordes, bbox_poste)
    densidad_categoria = clasificar_densidad(densidad)

    cables_min, cables_max, cables_estimado = estimar_rango_cables(picos_izq, picos_der, densidad_categoria)

    # NUEVO: si ya entrenaron un modelo con entrenar_modelo_cables.py,
    # se usa su prediccion como punto central del rango en vez del
    # punto central heuristico. El ANCHO del rango (que tan seguros
    # estamos) se sigue calculando exactamente igual que antes -- solo
    # cambia de donde sale el centro. Si no hay modelo entrenado
    # todavia, predecir_estimado_ml() devuelve None y no cambia nada.
    estimado_ml = predecir_estimado_ml(ancho_poste, picos_izq, picos_der, densidad)
    if estimado_ml is not None:
        ancho_rango = cables_max - cables_min
        cables_estimado = estimado_ml
        cables_min = max(0, estimado_ml - ancho_rango // 2)
        cables_max = estimado_ml + ancho_rango // 2

    tipo_previo = clasificar_tipo_poste(ancho_poste) if ancho_poste else "REVISION MANUAL"
    cables_min, cables_max = sanear_rango_cables(cables_min, cables_max, tipo_previo)
    # BUG CORREGIDO: antes evaluar_riesgo() recibia el conteo crudo
    # (num_cables), un numero DISTINTO al que terminaba mostrandose en
    # el rango (cables_min-cables_max, que ya incluye la correccion del
    # modelo ML si existe y el saneo por tipo de poste). Eso podia dar
    # un informe contradictorio: "entre 0 y 12 cables" pero "PELIGRO:
    # CONGESTION EXTREMA", porque el riesgo se decidia con un numero
    # que ni se le mostraba al usuario. Ahora ambos usan el mismo
    # numero central saneado (cables_estimado).
    cables_estimado = max(cables_min, min(cables_estimado, cables_max))

    # NUEVO: se calculan ANTES de decidir el riesgo (antes se calculaban
    # despues, asi que nunca podian usarse para confirmar si la densidad
    # alta era un nudo real o solo ruido localizado).
    riesgo_contacto_persona = detectar_riesgo_contacto_persona(bboxes_yolo, bbox_poste, contornos_cables)
    num_nudos_aprox = contar_nudos_cables(bordes, bbox_poste) if densidad_categoria in ("DENSO", "CRITICO") else 0
    cable_bajo_detectado = False

    # NUEVO: si hay internet y una API key de Roboflow configurada, se
    # consulta el workflow general (nudos + persona + cable colgando
    # bajo, todo en una sola llamada). Si no hay API key, no hay
    # internet, o falla por cualquier motivo, devuelve None y se sigue
    # usando lo que ya se calculo localmente -- nunca se cae la app
    # por esto.
    resultado_roboflow = analizar_con_workflow_general(img)
    if resultado_roboflow is not None:
        if resultado_roboflow.get("tangled-wires", 0) > 0:
            num_nudos_aprox = resultado_roboflow["tangled-wires"]
        if resultado_roboflow.get("human", 0) > 0:
            riesgo_contacto_persona = True
        cable_bajo_detectado = resultado_roboflow.get("low-hanging-wire", 0) > 0

    estado, color_bgr, tipo, recomendacion, nudo_detectado = evaluar_riesgo(
        cables_estimado, ancho_poste, densidad_categoria,
        num_nudos_aprox=num_nudos_aprox, riesgo_contacto_persona=riesgo_contacto_persona,
        cable_bajo_detectado=cable_bajo_detectado)

    mensaje = generar_mensaje_poste(
        tipo, cables_min, cables_max, estado, recomendacion, nudo_detectado,
        riesgo_contacto_persona=riesgo_contacto_persona, num_nudos_aprox=num_nudos_aprox,
        cable_bajo_detectado=cable_bajo_detectado)

    img_procesada = _dibujar_resultado(
        img_original, bboxes_yolo, bboxes_veg, bbox_poste, contornos_cables,
        tipo, cables_min, cables_max, densidad_categoria)

    postes = [{
        "tipo": tipo,
        "cables_min": cables_min,
        "cables_max": cables_max,
        "cables_estimado": cables_estimado,
        "estado": estado,
        "color_bgr": color_bgr,
        "recomendacion": recomendacion,
        "nudo_detectado": nudo_detectado,
        "num_nudos_aprox": num_nudos_aprox,
        "riesgo_contacto_persona": riesgo_contacto_persona,
        "cable_bajo_detectado": cable_bajo_detectado,
        "densidad_categoria": densidad_categoria,
        "mensaje": mensaje,
        "ancho_px": ancho_poste,
        "bbox": bbox_poste,
    }]

    return {
        "img_original": img_original,
        "img_procesada": img_procesada,
        "postes": postes,
    }


def resumen_texto(postes):
    if not postes:
        return "No se detecto ningun poste en la imagen.", True

    lineas = []
    hay_alerta = False
    for i, p in enumerate(postes, start=1):
        lineas.append(f"Poste {i}: {p['mensaje']}")
        if "PELIGRO" in p["estado"] or "ALERTA" in p["estado"]:
            hay_alerta = True

    return "\n\n".join(lineas), hay_alerta


def _dibujar_resultado(img, bboxes_yolo, bboxes_veg, bbox_poste, contornos_cables,
                        tipo, cables_min, cables_max, densidad_categoria=None):
    resultado = img.copy()

    for (x1, y1, x2, y2, clase) in bboxes_yolo:
        cv2.rectangle(resultado, (x1, y1), (x2, y2), COLOR_BASURA, 2)
        cv2.putText(resultado, clase, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BASURA, 2)

    for (x1, y1, x2, y2) in bboxes_veg:
        cv2.rectangle(resultado, (x1, y1), (x2, y2), COLOR_VEGETACION, 2)
        cv2.putText(resultado, "VEGETACION", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_VEGETACION, 2)

    if bbox_poste:
        px1, py1, px2, py2 = bbox_poste
        overlay = resultado.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), COLOR_POSTE, -1)
        cv2.addWeighted(overlay, 0.3, resultado, 0.7, 0, resultado)
        cv2.rectangle(resultado, (px1, py1), (px2, py2), COLOR_POSTE, 3)
        etiqueta = f"{tipo} | {cables_min}-{cables_max} cables"
        if densidad_categoria:
            etiqueta += f" | {densidad_categoria}"
        cv2.putText(resultado, etiqueta, (px1, max(20, py1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_POSTE, 2)

    if contornos_cables:
        cv2.drawContours(resultado, contornos_cables, -1, (0, 0, 0), 5)
        cv2.drawContours(resultado, contornos_cables, -1, COLOR_CABLES, 2)

    return resultado
