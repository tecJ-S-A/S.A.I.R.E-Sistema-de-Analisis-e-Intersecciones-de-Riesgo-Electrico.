import cv2
import numpy as np
from scipy.signal import find_peaks


# =====================================================================
# 1) DETECCION DE POSTE(S) POR CANDIDATOS (tamano, posicion, aspecto)
# =====================================================================
def detectar_postes_candidatos(bordes, max_candidatos=2):
    """Encuentra varios picos de masa vertical (candidatos a poste) y
    devuelve los mejores `max_candidatos`, puntuados por altura de
    senal, que tan centrados estan, y que tan 'postiforme' es su
    relacion alto/ancho (delgado y alto puntua mejor que ancho y bajo,
    que suele ser fachada de edificio)."""
    alto, ancho = bordes.shape[:2]

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (5, int(alto * 0.15)))
    verticales = cv2.morphologyEx(bordes, cv2.MORPH_OPEN, kernel_v)
    verticales = cv2.dilate(verticales, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10)))

    perfil = np.sum(verticales, axis=0).astype(float)
    margen = int(ancho * 0.1)
    perfil[:margen] = 0
    perfil[-margen:] = 0

    perfil_suave = cv2.GaussianBlur(perfil, (61, 1), 0).flatten()
    if len(perfil_suave) == 0 or perfil_suave.max() < 500:
        return []

    picos, _ = find_peaks(
        perfil_suave,
        height=perfil_suave.max() * 0.35,
        distance=max(10, int(ancho * 0.08)),
    )
    if len(picos) == 0:
        return []

    centro_imagen = ancho / 2.0
    picos_ordenados = sorted(int(p) for p in picos)
    candidatos = []
    for idx, x_centro in enumerate(picos_ordenados):
        max_val = float(perfil_suave[x_centro])

        # BUG CORREGIDO: antes el ancho a media altura podia "cruzar"
        # hasta el siguiente poste si no habia un valle profundo entre
        # ambos (ej: la maqueta con dos parantes de madera cercanos).
        # Eso fusionaba dos postes en un solo candidato clasificado
        # como un poste anormalmente ancho ("USO MIXTO" cuando en
        # realidad eran dos postes delgados). Ahora la busqueda del
        # ancho nunca puede pasar del punto medio hacia el candidato
        # vecino, sin importar que tan alto siga el perfil ahi.
        limite_izq = 0 if idx == 0 else (picos_ordenados[idx - 1] + x_centro) // 2
        limite_der = (ancho - 1) if idx == len(picos_ordenados) - 1 else (x_centro + picos_ordenados[idx + 1]) // 2

        umbral_mitad = max_val * 0.5
        izq = x_centro
        while izq > limite_izq and perfil_suave[izq] > umbral_mitad:
            izq -= 1
        der = x_centro
        while der < limite_der and perfil_suave[der] > umbral_mitad:
            der += 1
        ancho_poste = max(20, min(140, der - izq))

        relacion_aspecto = alto / ancho_poste
        dist_centro_norm = abs(x_centro - centro_imagen) / (ancho / 2.0)

        score = (
            max_val
            * (1.0 - 0.5 * dist_centro_norm)
            * min(1.3, 0.5 + relacion_aspecto / 15.0)
        )

        bbox = (
            max(0, x_centro - ancho_poste // 2), 0,
            min(ancho, x_centro + ancho_poste // 2), alto,
        )

        candidatos.append({
            "x_centro": int(x_centro),
            "ancho_px": int(ancho_poste),
            "bbox": bbox,
            "score": float(score),
        })

    candidatos.sort(key=lambda c: c["score"], reverse=True)
    return candidatos[:max_candidatos]


def detectar_estructura(bordes):
    """Compatibilidad hacia atras: devuelve solo el mejor candidato."""
    candidatos = detectar_postes_candidatos(bordes, max_candidatos=1)
    if not candidatos:
        return 0, None
    mejor = candidatos[0]
    return mejor["ancho_px"], mejor["bbox"]


# =====================================================================
# 2) CABLES: CONTEO ESTIMADO (rango, no numero exacto) + DENSIDAD
# =====================================================================
def _contar_picos_en_franja(bordes, x_inicio, x_fin, alto):
    franja = bordes[:, max(0, x_inicio):min(bordes.shape[1], x_fin)]
    if franja.shape[1] == 0:
        return 0, None, np.array([], dtype=int)

    ancho_real = franja.shape[1]
    perfil = np.sum(franja > 0, axis=1).astype(float)
    perfil_suave = cv2.GaussianBlur(perfil, (1, 7), 0).flatten()

    altura_minima = ancho_real * 0.55
    prominencia_minima = ancho_real * 0.12

    picos, _ = find_peaks(perfil_suave, height=altura_minima,
                           distance=8, prominence=prominencia_minima)

    return len(picos), perfil_suave, picos


def _combinar_conteos_lado(picos_izq, picos_der):
    lo, hi = min(picos_izq, picos_der), max(picos_izq, picos_der)
    if lo == 0 or lo < hi * 0.5:
        return hi
    return round((picos_izq + picos_der) / 2)


def detectar_cables_detallado(bordes, bbox_poste):
    """Igual que la version anterior de detectar_cables(), pero ademas
    devuelve los conteos de CADA lado por separado (picos_izq,
    picos_der) sin combinar todavia. Esto es lo que permite construir
    un RANGO estimado en vez de un numero unico: si ambos lados
    coinciden, el rango es angosto (alta confianza); si difieren
    mucho, el rango se ensancha (baja confianza, probablemente hay
    oclusion o un nudo dificil de leer desde un solo lado)."""
    alto, ancho = bordes.shape[:2]
    margen_poste = 20
    ancho_franja = 40

    cables = bordes.copy()
    ancho_hueco = 0
    if bbox_poste:
        x1, _, x2, _ = bbox_poste
        x1e, x2e = max(0, x1 - margen_poste), min(ancho, x2 + margen_poste)
        cv2.rectangle(cables, (x1e, 0), (x2e, alto), 0, -1)
        ancho_hueco = x2e - x1e

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
    cables = cv2.morphologyEx(cables, cv2.MORPH_OPEN, kernel_h)

    ancho_union = max(60, ancho_hueco + 40)
    kernel_unir = cv2.getStructuringElement(cv2.MORPH_RECT, (ancho_union, 5))
    cables_unidos = cv2.morphologyEx(cables, cv2.MORPH_CLOSE, kernel_unir)

    contornos, _ = cv2.findContours(cables_unidos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_validos = [c for c in contornos
                          if cv2.boundingRect(c)[2] > 50
                          and cv2.boundingRect(c)[2] / max(1, cv2.boundingRect(c)[3]) > 1.5]

    picos_izq, picos_der = 0, 0
    if bbox_poste:
        x1, _, x2, _ = bbox_poste
        franja_izq_fin = max(0, x1 - margen_poste)
        franja_izq_ini = max(0, franja_izq_fin - ancho_franja)
        franja_der_ini = min(ancho, x2 + margen_poste)
        franja_der_fin = min(ancho, franja_der_ini + ancho_franja)

        picos_izq, _, _ = _contar_picos_en_franja(bordes, franja_izq_ini, franja_izq_fin, alto)
        picos_der, _, _ = _contar_picos_en_franja(bordes, franja_der_ini, franja_der_fin, alto)

    conteo_por_picos = _combinar_conteos_lado(picos_izq, picos_der)
    conteo_final = conteo_por_picos if conteo_por_picos > 0 else len(contornos_validos)

    return max(0, conteo_final), contornos_validos, picos_izq, picos_der


def detectar_cables(bordes, bbox_poste):
    """Se mantiene para compatibilidad (evaluar_precision.py la sigue
    usando tal cual). Internamente llama a detectar_cables_detallado()
    y descarta el detalle por lado."""
    conteo, contornos, _, _ = detectar_cables_detallado(bordes, bbox_poste)
    return conteo, contornos


def estimar_rango_cables(picos_izq, picos_der, densidad_categoria=None):
    """NUEVO: en vez de un numero unico, entrega (minimo, maximo,
    estimado_central).

    BUG CORREGIDO: la version anterior fijaba la incertidumbre en
    abs(picos_izq - picos_der) SIN TECHO. Si un solo lado leia ruido de
    fondo (una mancha en la pizarra, un reflejo, una sombra) y contaba
    de mas, la discrepancia entre lados se disparaba (ej: 3 vs 22) y
    el rango final terminaba siendo algo inutil como "3 a 41 cables".
    Una discrepancia enorme entre lados es señal de que un lado esta
    fallando, no de que el rango real sea tan ancho.

    Ahora la incertidumbre crece con la discrepancia, pero nunca mas
    alla del 40% del valor estimado (minimo 2, maximo absoluto 6). Eso
    da rangos que se sienten como una estimacion razonable ("8 a 12
    cables") en vez de una rendicion ("entre 3 y 25")."""
    estimado = _combinar_conteos_lado(picos_izq, picos_der)
    diferencia = abs(picos_izq - picos_der)

    techo_incertidumbre = max(2, round(estimado * 0.4))
    incertidumbre = min(max(1, diferencia), techo_incertidumbre, 6)

    minimo = max(0, estimado - incertidumbre)
    maximo = estimado + incertidumbre
    if densidad_categoria in ("DENSO", "CRITICO"):
        maximo = int(round(maximo * 1.25))

    return minimo, maximo, estimado


def sanear_rango_cables(cables_min, cables_max, tipo):
    """NUEVO: ultima red de seguridad. Aunque estimar_rango_cables() ya
    acota la incertidumbre, esto pone un techo absoluto ligado al TIPO
    de poste detectado: no tiene sentido reportar 40 cables de rango
    superior en un poste que se clasifico como MEDIA TENSION (donde el
    limite de peligro real es 9). El techo es 3 veces el limite de
    peligro de ese tipo — bastante por encima de lo normal, pero ya
    no un numero arbitrario sin relacion con la realidad."""
    cfg = TIPOS_POSTE.get(tipo)
    if not cfg:
        return cables_min, cables_max

    techo_absoluto = cfg["peligro"] * 3
    cables_max = min(cables_max, techo_absoluto)
    cables_min = min(cables_min, cables_max)
    return cables_min, cables_max


def calcular_densidad_cables(bordes, bbox_poste, margen_poste=20, ancho_franja=40):
    """Fraccion de la franja lateral al poste cubierta por 'algo tipo
    cable'. No se rompe con nudos: cables bien separados o amontonados
    en el mismo espacio fisico dan ambos una densidad alta, que es la
    senal que de verdad importa para el riesgo."""
    alto, ancho = bordes.shape[:2]
    if not bbox_poste:
        return 0.0

    x1, _, x2, _ = bbox_poste
    franja_izq_fin = max(0, x1 - margen_poste)
    franja_izq_ini = max(0, franja_izq_fin - ancho_franja)
    franja_der_ini = min(ancho, x2 + margen_poste)
    franja_der_fin = min(ancho, franja_der_ini + ancho_franja)

    densidades = []
    for ini, fin in [(franja_izq_ini, franja_izq_fin), (franja_der_ini, franja_der_fin)]:
        franja = bordes[:, ini:fin]
        if franja.size == 0:
            continue
        densidades.append(np.count_nonzero(franja) / franja.size)

    return float(np.mean(densidades)) if densidades else 0.0


UMBRALES_DENSIDAD = {
    "LIGERO": 0.04,
    "MODERADO": 0.09,
    "DENSO": 0.16,
}


def clasificar_densidad(densidad):
    if densidad < UMBRALES_DENSIDAD["LIGERO"]:
        return "LIGERO"
    elif densidad < UMBRALES_DENSIDAD["MODERADO"]:
        return "MODERADO"
    elif densidad < UMBRALES_DENSIDAD["DENSO"]:
        return "DENSO"
    else:
        return "CRITICO"


def contar_cables_hough(bordes, bbox_poste, alto, ancho):
    """Sin cambios: se mantiene para evaluar_precision.py."""
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=40,
                              minLineLength=50, maxLineGap=25)
    if lineas is None:
        return 0, []

    if bbox_poste:
        x1p, _, x2p, _ = bbox_poste
        x_ref = x1p - 20 if x1p > ancho / 2 else x2p + 20
        x_ref = max(0, min(ancho - 1, x_ref))
    else:
        x_ref = ancho // 2

    tolerancia = 30
    segmentos_validos = []
    cruces_y = []

    for linea in lineas.reshape(-1, 4):
        x1, y1, x2, y2 = linea
        if x1 == x2:
            continue
        angulo = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angulo) > 35:
            continue
        x_izq, x_der = min(x1, x2), max(x1, x2)
        if not (x_izq - tolerancia <= x_ref <= x_der + tolerancia):
            continue

        pendiente = (y2 - y1) / (x2 - x1)
        y_en_ref = y1 + pendiente * (x_ref - x1)
        if 0 <= y_en_ref <= alto:
            cruces_y.append(y_en_ref)
            segmentos_validos.append((x1, y1, x2, y2))

    if not cruces_y:
        return 0, segmentos_validos

    cruces_y.sort()
    grupos = [[cruces_y[0]]]
    for y in cruces_y[1:]:
        if y - grupos[-1][-1] <= 10:
            grupos[-1].append(y)
        else:
            grupos.append([y])

    return len(grupos), segmentos_validos


# =====================================================================
# 3) TIPO DE POSTE Y RIESGO — con los rangos reales investigados en Peru
# =====================================================================
# Fuente: investigacion propia del equipo sobre tipos de poste en Peru
# (altura y espesor de pared de concreto por tipo). Como de una sola
# foto no se puede medir la altura real sin un objeto de referencia,
# se usa el ANCHO EN PIXELES del poste detectado como proxy visual del
# tipo (mas grueso en la foto = generalmente mas robusto = mayor
# capacidad de cables). Es una aproximacion razonable para un MVP de
# hackathon, no una medicion certificada — para afinarla de verdad
# hace falta calibrar los puntos de corte (ancho_min_px/ancho_max_px)
# con fotos propias de postes de tipo conocido, el mismo espiritu que
# evaluar_precision.py.
TIPOS_POSTE = {
    # AJUSTE: los umbrales anteriores (MEDIA TENSION hasta 60px,
    # seguro=6) hacian que fotos de postes normales de BAJA TENSION
    # cayeran por error en el balde de MEDIA TENSION -- mucho mas
    # estricto (seguro=6) -- y disparaban PELIGRO con apenas 10 cables,
    # algo completamente normal en la calle peruana. Ahora MEDIA
    # TENSION exige un poste realmente angosto en la foto (rango de
    # pixeles mas chico) para clasificarse asi, y su techo "seguro"
    # sube un poco (de 6 a 8) para no ser tan gatillo facil incluso
    # cuando SI es un poste delgado real. BAJA TENSION (el caso mas
    # comun en la calle) ahora cubre un rango de ancho mas amplio por
    # defecto, y su "seguro" sube a 20 -- por encima de los ~15 cables
    # que ustedes mismos observaron como normales en Chiclayo, para
    # que ese caso tipico no dispare advertencias todo el tiempo.
    "MEDIA TENSION": {"ancho_min_px": 0,   "ancho_max_px": 35,  "seguro": 8,  "peligro": 14},
    "BAJA TENSION":  {"ancho_min_px": 35,  "ancho_max_px": 100, "seguro": 20, "peligro": 30},
    "USO MIXTO":     {"ancho_min_px": 100, "ancho_max_px": 9999, "seguro": 55, "peligro": 70},
}


def clasificar_tipo_poste(ancho_poste):
    for tipo, cfg in TIPOS_POSTE.items():
        if cfg["ancho_min_px"] <= ancho_poste < cfg["ancho_max_px"]:
            return tipo
    return "USO MIXTO"


_MAPA_DENSIDAD_A_NIVEL = {"LIGERO": 0, "MODERADO": 1, "DENSO": 2, "CRITICO": 3}

_COLORES_NIVEL = {
    0: (0, 255, 0),
    1: (0, 165, 255),
    2: (0, 140, 255),
    3: (0, 0, 255),
}

_TEXTOS_NIVEL = {
    0: "ESTABLE: DENTRO DE LA NORMA",
    1: "ADVERTENCIA: CONGESTION MODERADA",
    2: "ADVERTENCIA: CONGESTION ALTA",
    3: "PELIGRO: CONGESTION EXTREMA",
}

RECOMENDACIONES = {
    0: "Sin accion inmediata. Continuar con monitoreo periodico.",
    1: "Vigilar este poste y programar una revision en las proximas semanas.",
    2: "Se recomienda reportar a la empresa electrica en los proximos dias.",
    3: "Riesgo alto: reportar de inmediato a la empresa electrica. Evitar permanecer cerca del poste.",
}


def evaluar_riesgo(num_cables, ancho_poste, densidad_categoria=None, num_nudos_aprox=0,
                    riesgo_contacto_persona=False, cable_bajo_detectado=False):
    """CAMBIO IMPORTANTE: ahora devuelve una tupla de 5 elementos en
    vez de 3:
        (estado, color_bgr, tipo, recomendacion, nudo_detectado)

    - estado / color_bgr / tipo: igual que antes en espiritu, pero
      "tipo" ahora usa las categorias reales investigadas (MEDIA
      TENSION / BAJA TENSION / USO MIXTO) y sus rangos de cables.
    - recomendacion: texto listo para mostrar al usuario o meter en
      un PDF, no un dato tecnico.
    - nudo_detectado: True si la densidad indica un amontonamiento
      DENSO o CRITICO (posible conexion no regulada) — se reporta
      aparte porque es un tipo de alerta distinto (fiscalizacion),
      no solo "hay muchos cables".

    NUEVO en num_nudos_aprox / riesgo_contacto_persona: la densidad ya
    no esta topada a ciegas. Se distinguen dos situaciones distintas
    que antes se trataban igual:

    1. Densidad alta pero UN SOLO punto de amontonamiento
       (num_nudos_aprox <= 1) y sin persona cerca: probablemente sea
       ruido localizado (una caja de conexiones, el soporte de un
       farol pegado al poste) -- se mantiene el tope de +1 nivel.
    2. Densidad alta CONFIRMADA por varios puntos de amontonamiento
       (num_nudos_aprox >= 2) o por una persona trabajando cerca: esto
       ya es evidencia real de un nudo genuino, no ruido -- se permite
       escalar el riesgo sin tope, hasta PELIGRO si corresponde.

    Solo pipeline.py llama a esta funcion, asi que este cambio de
    firma no rompe evaluar_precision.py ni main.py.
    """
    if ancho_poste == 0:
        return (
            "ALERTA: POSTE NO IDENTIFICADO", (0, 0, 255), "REVISION MANUAL",
            "No se pudo identificar un poste en la imagen. Repite la toma "
            "con el poste mas centrado y visible.",
            False,
        )

    tipo = clasificar_tipo_poste(ancho_poste)
    cfg = TIPOS_POSTE[tipo]

    if num_cables <= cfg["seguro"]:
        nivel_conteo = 0
    elif num_cables <= cfg["peligro"]:
        nivel_conteo = 1
    else:
        nivel_conteo = 3

    nivel_densidad = _MAPA_DENSIDAD_A_NIVEL.get(densidad_categoria) if densidad_categoria else None
    if nivel_densidad is not None:
        evidencia_confirmada = num_nudos_aprox >= 2 or riesgo_contacto_persona
        if evidencia_confirmada:
            nivel_final = max(nivel_conteo, nivel_densidad)
        else:
            nivel_final = max(nivel_conteo, min(nivel_densidad, nivel_conteo + 1))
    else:
        nivel_final = nivel_conteo

    # NUEVO: un cable colgando bajo (a la altura de una persona o un
    # vehiculo) es un peligro por si solo, independiente de cuantos
    # cables haya en total o que tan denso este el amasijo -- por eso
    # garantiza un piso minimo de riesgo (ADVERTENCIA ALTA) sin
    # importar lo que digan los demas indicadores.
    if cable_bajo_detectado:
        nivel_final = max(nivel_final, 2)

    nudo_detectado = densidad_categoria in ("DENSO", "CRITICO")

    return (
        _TEXTOS_NIVEL[nivel_final], _COLORES_NIVEL[nivel_final], tipo,
        RECOMENDACIONES[nivel_final], nudo_detectado,
    )


def generar_mensaje_poste(tipo, cables_min, cables_max, estado, recomendacion, nudo_detectado,
                           riesgo_contacto_persona=False, num_nudos_aprox=0, cable_bajo_detectado=False):
    """Arma el mensaje en lenguaje llano que ve el usuario final (y que
    tambien se reutiliza tal cual dentro del PDF). Nada de numeros
    tecnicos sueltos: tipo de poste, rango estimado, estado y que
    hacer al respecto."""
    partes = [
        f"Tipo de poste estimado: {tipo}",
        f"Cables detectados: entre {cables_min} y {cables_max} (estimado)",
        f"Estado: {estado}",
        f"Recomendacion: {recomendacion}",
    ]
    if nudo_detectado:
        extra_nudo = ""
        if num_nudos_aprox > 0:
            extra_nudo = f" Se detectaron aproximadamente {num_nudos_aprox} punto(s) de amontonamiento."
        partes.append(
            "Se detecto un amontonamiento denso de cables (posible conexion "
            "no regulada)." + extra_nudo + " Ademas de mantenimiento, esto "
            "conviene reportarlo como caso de fiscalizacion. Se sugiere "
            "tambien pedir a la empresa electrica que verifique si hay "
            "cables aereos en desuso en esta zona -- es comun que amasijos "
            "como este incluyan cableado de proveedores anteriores que "
            "nunca se retiro, y su remocion reduce el riesgo sin afectar "
            "servicio activo. Esto no lo puede determinar la foto por si "
            "sola, es una verificacion que le corresponde hacer a la "
            "empresa con sus propios registros."
        )
    if riesgo_contacto_persona:
        partes.append(
            "ALERTA DE SEGURIDAD: se detecto una persona a corta distancia "
            "del poste o los cables en la foto. Verificar si hay riesgo de "
            "contacto directo y, de ser asi, alejarse y reportar de "
            "inmediato."
        )
    if cable_bajo_detectado:
        partes.append(
            "ALERTA: se detecto un cable colgando a baja altura, a nivel de "
            "una persona o un vehiculo. Esto es peligroso independientemente "
            "de cuantos cables haya en total -- reportar de inmediato."
        )
    return "\n".join(partes)


# =====================================================================
# 4) SENALES ADICIONALES DEL BANCO DE IMAGENES: PERSONAS Y NUDOS
# =====================================================================
def detectar_riesgo_contacto_persona(bboxes_yolo, bbox_poste, contornos_cables, margen=40):
    """Revisa si alguna persona detectada por YOLO (ya viene en
    bboxes_yolo, que vision_ia.py entrega para dibujar en la UI) esta a
    corta distancia del poste o de la zona donde se detectaron cables.

    IMPORTANTE sobre lo que esto SI y NO puede afirmar: una foto en 2D
    no permite saber con certeza si una persona esta tocando un cable
    (la profundidad se pierde). Lo que si se puede detectar de forma
    confiable es proximidad en la imagen -- que ya es una señal util,
    como el caso de su banco de imagenes donde la persona estira la
    mano hacia el cableado. Por eso el mensaje se redacta como "alerta
    de proximidad a verificar", no como una afirmacion de contacto.
    """
    personas = [(x1, y1, x2, y2) for (x1, y1, x2, y2, clase) in bboxes_yolo if clase.upper() == "PERSON"]
    if not personas:
        return False

    zona_riesgo = None
    if bbox_poste:
        x1, y1, x2, y2 = bbox_poste
        zona_riesgo = [x1 - margen, y1, x2 + margen, y2]

    for cnt in contornos_cables:
        x, y, w, h = cv2.boundingRect(cnt)
        caja = [x - margen, y - margen, x + w + margen, y + h + margen]
        if zona_riesgo is None:
            zona_riesgo = caja
        else:
            zona_riesgo = [
                min(zona_riesgo[0], caja[0]), min(zona_riesgo[1], caja[1]),
                max(zona_riesgo[2], caja[2]), max(zona_riesgo[3], caja[3]),
            ]

    if zona_riesgo is None:
        return False

    for (px1, py1, px2, py2) in personas:
        if px1 < zona_riesgo[2] and px2 > zona_riesgo[0] and py1 < zona_riesgo[3] and py2 > zona_riesgo[1]:
            return True
    return False


def contar_nudos_cables(bordes, bbox_poste, margen_zona=150, tamano_bloque=40, umbral_bloque=0.5):
    """Aproxima CUANTOS puntos de amontonamiento denso ('nudos') hay
    cerca del poste, contando regiones conectadas donde la densidad
    local de pixeles tipo-cable supera un umbral por bloques.

    Esto es una aproximacion geometrica, no la precision de un ojo
    humano marcando circulos como en su banco de imagenes -- pero da
    una cantidad orientativa util para el informe ('se detectaron
    aprox. 3 puntos de amontonamiento') sin necesitar entrenar nada."""
    if not bbox_poste:
        return 0

    alto, ancho = bordes.shape[:2]
    x1, _, x2, _ = bbox_poste
    zona_x1 = max(0, x1 - margen_zona)
    zona_x2 = min(ancho, x2 + margen_zona)
    zona = bordes[:, zona_x1:zona_x2]
    if zona.size == 0:
        return 0

    alto_zona, ancho_zona = zona.shape
    mapa_densidad = np.zeros_like(zona, dtype=np.uint8)

    for by in range(0, alto_zona, tamano_bloque):
        for bx in range(0, ancho_zona, tamano_bloque):
            bloque = zona[by:by + tamano_bloque, bx:bx + tamano_bloque]
            if bloque.size == 0:
                continue
            densidad_bloque = np.count_nonzero(bloque) / bloque.size
            if densidad_bloque > umbral_bloque:
                mapa_densidad[by:by + tamano_bloque, bx:bx + tamano_bloque] = 255

    num_etiquetas, _ = cv2.connectedComponents(mapa_densidad)
    return max(0, num_etiquetas - 1)  # la etiqueta 0 siempre es el fondo
