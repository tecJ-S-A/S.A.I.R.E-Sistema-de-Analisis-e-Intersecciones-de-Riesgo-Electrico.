import os
import json
import time
import uuid
import cv2

CARPETA_HISTORIAL = "historial"
ARCHIVO_REGISTROS = os.path.join(CARPETA_HISTORIAL, "registros.json")
MAX_REGISTROS = 50


def _asegurar_carpeta():
    os.makedirs(CARPETA_HISTORIAL, exist_ok=True)


def cargar_historial():
    _asegurar_carpeta()
    if not os.path.exists(ARCHIVO_REGISTROS):
        return []
    try:
        with open(ARCHIVO_REGISTROS, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def _guardar_json(registros):
    with open(ARCHIVO_REGISTROS, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)


def guardar_registro(img_procesada, tipo, cables_min, cables_max, estado,
                      recomendacion="", zona="", notas_usuario="",
                      notas_pulidas="", nudo_detectado=False,
                      num_nudos_aprox=0, riesgo_contacto_persona=False):
    """CAMBIO DE FIRMA respecto a la version anterior: donde antes iba
    un solo `num_cables`, ahora van `cables_min` y `cables_max` (el
    rango estimado que entrega el pipeline). Los parametros nuevos
    (recomendacion, zona, notas_usuario, notas_pulidas, nudo_detectado)
    son todos con default, asi que si en algun lugar todavia se llama
    con menos argumentos no revienta — pero para aprovechar el reporte
    en PDF y el resumen conviene pasarlos todos desde main.py.

    zona: texto libre tipo "Urb. Latina, Chiclayo" o el nombre del
    distrito — no es GPS real, es la ubicacion aproximada que el
    usuario confirma antes de tomar la foto (ver _pedir_zona_inicial en
    main.py). notas_usuario / notas_pulidas: la nota de campo que
    escribe el usuario y su version redactada (ver notas_ia.py).
    """
    _asegurar_carpeta()

    marca_tiempo = time.strftime("%Y%m%d_%H%M%S")
    sufijo = uuid.uuid4().hex[:6]
    ruta_miniatura = os.path.join(CARPETA_HISTORIAL, f"{marca_tiempo}_{sufijo}.jpg")
    cv2.imwrite(ruta_miniatura, cv2.resize(img_procesada, (160, 120)))

    registros = cargar_historial()
    registros.insert(0, {
        "fecha": time.strftime("%d/%m/%Y %H:%M"),
        "miniatura": ruta_miniatura,
        "tipo": tipo,
        "cables_min": cables_min,
        "cables_max": cables_max,
        "estado": estado,
        "recomendacion": recomendacion,
        "zona": zona,
        "notas_usuario": notas_usuario,
        "notas_pulidas": notas_pulidas,
        "nudo_detectado": nudo_detectado,
        "num_nudos_aprox": num_nudos_aprox,
        "riesgo_contacto_persona": riesgo_contacto_persona,
    })

    if len(registros) > MAX_REGISTROS:
        sobrantes = registros[MAX_REGISTROS:]
        registros = registros[:MAX_REGISTROS]
        for r in sobrantes:
            _borrar_miniatura(r.get("miniatura"))

    _guardar_json(registros)
    return registros


def _borrar_miniatura(ruta_miniatura):
    if ruta_miniatura and os.path.exists(ruta_miniatura):
        try:
            os.remove(ruta_miniatura)
        except OSError:
            pass


def eliminar_registro(indice):
    registros = cargar_historial()
    if 0 <= indice < len(registros):
        _borrar_miniatura(registros[indice].get("miniatura"))
        registros.pop(indice)
        _guardar_json(registros)
    return registros


def vaciar_historial():
    registros = cargar_historial()
    for r in registros:
        _borrar_miniatura(r.get("miniatura"))
    _guardar_json([])
    return []


def resumen_historial():
    """NUEVO: es literalmente el "casi-reporte" para la empresa
    electrica sin tener que construir nada nuevo de cero — solo lee lo
    que ya esta guardado y lo agrupa. Devuelve:
        {
            "total": int,
            "por_estado": {"PELIGRO: ...": 5, "ESTABLE: ...": 12, ...},
            "por_zona": {"Urb. Latina": 3, "Sin zona": 2, ...},
            "con_nudo": int,
        }
    Se usa tal cual desde main.py (boton "Resumen") y desde
    reportes.py (para el PDF agregado).
    """
    registros = cargar_historial()
    por_estado = {}
    por_zona = {}
    con_nudo = 0

    for r in registros:
        estado = r.get("estado", "?")
        por_estado[estado] = por_estado.get(estado, 0) + 1

        zona = r.get("zona") or "Sin zona registrada"
        por_zona[zona] = por_zona.get(zona, 0) + 1

        if r.get("nudo_detectado"):
            con_nudo += 1

    return {
        "total": len(registros),
        "por_estado": por_estado,
        "por_zona": por_zona,
        "con_nudo": con_nudo,
    }
