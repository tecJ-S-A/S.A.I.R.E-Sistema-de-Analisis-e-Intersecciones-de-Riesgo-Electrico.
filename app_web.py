"""
app_web.py
Version web de S.A.I.R.E. PRO con Streamlit. Reutiliza EXACTAMENTE la
misma logica de siempre (pipeline.py, historial.py, reportes.py,
notas_ia.py). Tiene DOS modos:

  1. "Analizar una foto" -- el uso normal: subes una foto y la analiza.
  2. "Modo de validacion" -- carga uno de los 18 casos de tu banco de
     imagenes (los que estan en postes_test/) y compara lo que la IA
     calcula HOY contra la respuesta correcta que ya verificaron a
     mano (casos_referencia.json). Sirve para mostrarle al jurado que
     tan bien funciona el sistema, con evidencia, no solo decirlo.

INSTALAR:
    pip install streamlit

CORRER:
    streamlit run app_web.py
"""

import os
import json
import time

import cv2
import numpy as np
import streamlit as st

from pipeline import ejecutar_pipeline
from historial import guardar_registro, resumen_historial
from notas_ia import pulir_nota
from vision_ia import NOMBRE_MODELO_ACTIVO, CONF_POR_DEFECTO

CARPETA_CASOS = "postes_test"
ARCHIVO_REFERENCIA = os.path.join(CARPETA_CASOS, "casos_referencia.json")

st.set_page_config(page_title="S.A.I.R.E. PRO", page_icon="⚡", layout="wide")
st.title("⚡ S.A.I.R.E. PRO")
st.caption("Auditoria Inteligente de Postes y Cables")

with st.sidebar:
    st.subheader("Configuracion")
    zona = st.text_input("Zona / distrito", placeholder="Ej: La Victoria, Chiclayo")
    conf = st.slider("Sensibilidad IA", 0.10, 0.80, float(CONF_POR_DEFECTO), 0.01)
    st.caption(f"Motor de IA: {NOMBRE_MODELO_ACTIVO}")
    st.divider()
    modo = st.radio("Modo", ["Analizar una foto", "Modo de validacion (casos conocidos)"])
    st.divider()
    resumen = resumen_historial()
    st.metric("Postes auditados", resumen["total"])


def mostrar_resultado(resultado, poste, clave_widget):
    col_izq, col_der = st.columns(2)
    with col_izq:
        st.subheader("Original")
        st.image(cv2.cvtColor(resultado["img_original"], cv2.COLOR_BGR2RGB))
    with col_der:
        st.subheader("Deteccion de IA")
        st.image(cv2.cvtColor(resultado["img_procesada"], cv2.COLOR_BGR2RGB))

    color = "red" if ("PELIGRO" in poste["estado"] or "ALERTA" in poste["estado"]) else \
            "orange" if "ADVERTENCIA" in poste["estado"] else "green"
    st.markdown(f"### :{color}[{poste['estado']}]")
    st.write(poste["mensaje"].replace("\n", "\n\n"))

    if poste.get("riesgo_contacto_persona"):
        st.warning("⚠️ Se detecto una persona a corta distancia del poste o los cables.")

    return st.text_area("Nota de campo (opcional)", key=f"nota_{clave_widget}")


# ---------------------------------------------------------------
# MODO 1: analizar una foto subida por el usuario (galeria)
# ---------------------------------------------------------------
if modo == "Analizar una foto":
    archivo = st.file_uploader("Selecciona una foto (de tu galeria o un archivo)",
                                type=["jpg", "jpeg", "png"])
    if archivo is not None:
        datos = np.frombuffer(archivo.read(), np.uint8)
        img_bruta = cv2.imdecode(datos, cv2.IMREAD_COLOR)

        if img_bruta is None:
            st.error("No se pudo abrir esa imagen.")
        else:
            with st.spinner("Analizando imagen..."):
                resultado = ejecutar_pipeline(img_bruta, conf=conf)
                poste = resultado["postes"][0]

            nota_usuario = mostrar_resultado(resultado, poste, "subida")

            if st.button("Guardar en el historial", type="primary"):
                notas_pulidas = pulir_nota(nota_usuario) if nota_usuario else ""
                guardar_registro(
                    resultado["img_procesada"], poste["tipo"], poste["cables_min"], poste["cables_max"],
                    poste["estado"], recomendacion=poste["recomendacion"], zona=zona,
                    notas_usuario=nota_usuario, notas_pulidas=notas_pulidas,
                    nudo_detectado=poste["nudo_detectado"],
                    num_nudos_aprox=poste.get("num_nudos_aprox", 0),
                    riesgo_contacto_persona=poste.get("riesgo_contacto_persona", False),
                )
                st.success("Guardado en el historial.")

            if st.button("Generar informe PDF"):
                try:
                    import reportes
                    os.makedirs("temp", exist_ok=True)
                    ruta_original = os.path.join("temp", "web_original.jpg")
                    ruta_procesada = os.path.join("temp", "web_procesada.jpg")
                    cv2.imwrite(ruta_original, resultado["img_original"])
                    cv2.imwrite(ruta_procesada, resultado["img_procesada"])
                    registro = {
                        "tipo": poste["tipo"], "cables_min": poste["cables_min"],
                        "cables_max": poste["cables_max"], "estado": poste["estado"],
                        "recomendacion": poste["recomendacion"], "descripcion": poste["mensaje"],
                        "zona": zona, "notas_usuario": nota_usuario,
                        "notas_pulidas": pulir_nota(nota_usuario) if nota_usuario else "",
                        "nudo_detectado": poste["nudo_detectado"],
                        "num_nudos_aprox": poste.get("num_nudos_aprox", 0),
                        "riesgo_contacto_persona": poste.get("riesgo_contacto_persona", False),
                        "fecha": time.strftime("%d/%m/%Y %H:%M"),
                    }
                    ruta_pdf = reportes.generar_pdf_individual(
                        registro, ruta_imagen_procesada=ruta_procesada, ruta_imagen_original=ruta_original)
                    with open(ruta_pdf, "rb") as f:
                        st.download_button("Descargar informe PDF", f, file_name=os.path.basename(ruta_pdf))
                except ImportError:
                    st.error("Falta instalar fpdf2. Ejecuta: pip install fpdf2")
    else:
        st.info("Sube una foto para empezar el analisis.")

# ---------------------------------------------------------------
# MODO 2: validacion contra el banco de casos conocidos
# ---------------------------------------------------------------
else:
    if not os.path.exists(ARCHIVO_REFERENCIA):
        st.error(f"No encuentro {ARCHIVO_REFERENCIA}. Revisa el paso 2 de las instrucciones.")
    else:
        with open(ARCHIVO_REFERENCIA, "r", encoding="utf-8") as f:
            referencia = json.load(f)

        archivos_disponibles = [
            nombre for nombre in referencia
            if os.path.exists(os.path.join(CARPETA_CASOS, nombre))
        ]
        if not archivos_disponibles:
            st.warning(f"casos_referencia.json existe, pero no encuentro las fotos "
                       f"(caso01.jpg, etc.) dentro de {CARPETA_CASOS}/.")
        else:
            elegido = st.selectbox("Elige un caso conocido", archivos_disponibles)
            ruta_foto = os.path.join(CARPETA_CASOS, elegido)
            img_bruta = cv2.imread(ruta_foto)

            with st.spinner("Analizando imagen..."):
                resultado = ejecutar_pipeline(img_bruta, conf=conf)
                poste = resultado["postes"][0]

            mostrar_resultado(resultado, poste, elegido)

            correcto = referencia[elegido]
            st.divider()
            st.subheader("Comparacion con la respuesta validada")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rango IA (cables)", f"{poste['cables_min']}-{poste['cables_max']}")
            c1.metric("Rango correcto", f"{correcto['cables_min']}-{correcto['cables_max']}")
            c2.metric("Nudos IA (aprox.)", poste.get("num_nudos_aprox", 0))
            c2.metric("Nudos reales", correcto.get("nudos_reales", 0))
            c3.metric("Persona IA", "Si" if poste.get("riesgo_contacto_persona") else "No")
            c3.metric("Persona real", "Si" if correcto.get("persona") else "No")
            if correcto.get("nota"):
                c4.info(correcto["nota"])
