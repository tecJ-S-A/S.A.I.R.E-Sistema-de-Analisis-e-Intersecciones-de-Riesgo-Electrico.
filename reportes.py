"""
reportes.py
Genera los PDF que el usuario descarga y adjunta a un correo o
WhatsApp para la empresa electrica o la municipalidad. Usa fpdf2:

    pip install fpdf2

Dos funciones:
  - generar_pdf_individual(): un poste, con foto ORIGINAL y foto
    PROCESADA, caso identificado por nombre/ID, fecha, hora, zona,
    descripcion de lo detectado, recomendacion y comentarios del
    usuario.
  - generar_pdf_resumen(): el historial completo agrupado (para
    presionar con volumen: "15 postes en estado critico en tal zona").
"""

import os
import time
import uuid

from fpdf import FPDF

CARPETA_REPORTES = "reportes"


def _asegurar_carpeta():
    os.makedirs(CARPETA_REPORTES, exist_ok=True)


def generar_id_caso():
    """ID corto y legible para identificar el caso en el PDF, en el
    nombre del archivo, y en cualquier correo/whatsapp que se mande
    despues. Ej: SAIRE-20260721-4f2a"""
    return f"SAIRE-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:4]}"


def _separar_fecha_hora(fecha_texto):
    """El registro guarda 'fecha' como '21/07/2026 14:35'. Esto la
    separa en (fecha, hora) para mostrarlas como campos independientes
    en el PDF, que es como pidieron que se vea."""
    fecha_texto = fecha_texto or time.strftime("%d/%m/%Y %H:%M")
    partes = fecha_texto.split(" ")
    if len(partes) == 2:
        return partes[0], partes[1]
    return fecha_texto, "-"


def generar_pdf_individual(registro, ruta_imagen_procesada=None, ruta_imagen_original=None,
                            nombre_caso=None):
    """registro: dict con tipo, cables_min, cables_max, estado,
    recomendacion, zona, notas_usuario, notas_pulidas, nudo_detectado,
    fecha (y opcionalmente descripcion).

    ruta_imagen_procesada / ruta_imagen_original: rutas a las imagenes
    a full resolucion. Si solo se tiene una (por ejemplo al generar el
    PDF desde una fila vieja del historial, donde solo queda la
    miniatura), se usa esa nomas.

    nombre_caso: si no se pasa, se genera un ID automatico (ver
    generar_id_caso()) para que cada informe quede identificable aunque
    el usuario no le haya puesto nombre.

    Devuelve la ruta del PDF generado.
    """
    _asegurar_carpeta()

    caso_id = nombre_caso.strip() if nombre_caso else generar_id_caso()
    fecha, hora = _separar_fecha_hora(registro.get("fecha"))

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "S.A.I.R.E. PRO - Informe de Auditoria", ln=True)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 7, f"Caso: {caso_id}", ln=True)
    pdf.set_text_color(20, 20, 20)
    pdf.ln(2)

    # --- Datos generales ---
    pdf.set_font("Helvetica", "", 11)
    datos_generales = [
        ("Fecha", fecha),
        ("Hora", hora),
        ("Lugar / zona", registro.get("zona") or "No registrada"),
    ]
    for etiqueta, valor in datos_generales:
        pdf.cell(0, 7, f"{etiqueta}: {valor}", ln=True)
    pdf.ln(3)

    # --- Foto original (la que tomo la persona, sin marcas) ---
    if ruta_imagen_original and os.path.exists(ruta_imagen_original):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Foto original", ln=True)
        pdf.image(ruta_imagen_original, w=110)
        pdf.ln(4)

    # --- Foto procesada (con las detecciones marcadas) ---
    ruta_procesada = ruta_imagen_procesada or registro.get("miniatura")
    if ruta_procesada and os.path.exists(ruta_procesada):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Foto con deteccion de IA", ln=True)
        pdf.image(ruta_procesada, w=110)
        pdf.ln(4)

    # --- Estado y descripcion de lo detectado ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"Estado: {registro.get('estado', '?')}", ln=True)

    pdf.set_font("Helvetica", "", 11)
    descripcion = registro.get("descripcion") or (
        f"Se identifico un poste de tipo {registro.get('tipo', '?')} con un "
        f"estimado de {registro.get('cables_min', '?')} a "
        f"{registro.get('cables_max', '?')} cables (rango estimado por "
        "vision por computadora, no un conteo certificado)."
    )
    pdf.multi_cell(0, 6, descripcion)
    pdf.ln(2)

    if registro.get("nudo_detectado"):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(200, 60, 0)
        texto_nudo = ("ALERTA ADICIONAL: se detecto un amontonamiento denso "
                      "de cables")
        if registro.get("num_nudos_aprox"):
            texto_nudo += f" (aprox. {registro['num_nudos_aprox']} punto(s))"
        texto_nudo += (". Posible conexion no regulada - se sugiere "
                       "tratarlo tambien como caso de fiscalizacion, no solo "
                       "de mantenimiento. Se recomienda ademas solicitar a la "
                       "empresa electrica que verifique si hay cableado aereo "
                       "en desuso en esta zona (comun en amasijos de este tipo, "
                       "de proveedores anteriores nunca retirados); esto no lo "
                       "determina la foto, requiere revision de sus registros.")
        pdf.multi_cell(0, 6, texto_nudo)
        pdf.set_text_color(20, 20, 20)
        pdf.ln(2)

    if registro.get("riesgo_contacto_persona"):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(200, 0, 0)
        pdf.multi_cell(0, 6, "ALERTA DE SEGURIDAD: se detecto una persona a corta "
                              "distancia del poste o los cables en la foto. Esto "
                              "no confirma contacto directo (no se puede saber con "
                              "certeza desde una foto en 2D), pero amerita "
                              "verificacion y reporte prioritario.")
        pdf.set_text_color(20, 20, 20)
        pdf.ln(2)

    # --- Recomendacion ---
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Recomendacion:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, registro.get("recomendacion", "-"))
    pdf.ln(2)

    # --- Comentarios del usuario (ya redactados, ver notas_ia.py) ---
    comentarios = registro.get("notas_pulidas") or registro.get("notas_usuario")
    if comentarios:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Comentarios de quien reporta:", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, comentarios)
        pdf.ln(2)

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, "Informe generado automaticamente por S.A.I.R.E. PRO. Los "
                         "valores de cables son estimados por vision por computadora, "
                         "no un conteo certificado. Se recomienda verificacion en "
                         "campo antes de una intervencion.")

    marca = time.strftime("%Y%m%d_%H%M%S")
    ruta_salida = os.path.join(CARPETA_REPORTES, f"{caso_id}_{marca}.pdf")
    pdf.output(ruta_salida)
    return ruta_salida


def generar_pdf_resumen(registros, resumen, nombre_caso=None):
    """registros: lista completa de historial.cargar_historial().
    resumen: el dict que devuelve historial.resumen_historial().
    """
    _asegurar_carpeta()
    caso_id = nombre_caso.strip() if nombre_caso else generar_id_caso()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "S.A.I.R.E. PRO - Resumen de Auditorias", ln=True)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 7, f"Reporte: {caso_id}", ln=True)
    pdf.cell(0, 6, f"Generado: {time.strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.cell(0, 6, f"Total de postes auditados: {resumen.get('total', 0)}", ln=True)
    pdf.cell(0, 6, f"Casos con posible conexion no regulada: {resumen.get('con_nudo', 0)}", ln=True)
    pdf.ln(4)

    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Por estado de riesgo", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for estado, cantidad in resumen.get("por_estado", {}).items():
        pdf.cell(0, 7, f"- {estado}: {cantidad}", ln=True)

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Por zona", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for zona, cantidad in resumen.get("por_zona", {}).items():
        pdf.cell(0, 7, f"- {zona}: {cantidad}", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Detalle de casos criticos", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for r in registros:
        if "PELIGRO" in r.get("estado", "") or "ALERTA" in r.get("estado", ""):
            fecha, hora = _separar_fecha_hora(r.get("fecha"))
            linea = f"{fecha} {hora} | {r.get('zona') or 'sin zona'} | {r.get('tipo','?')} | {r.get('estado','?')}"
            pdf.multi_cell(0, 6, linea)

    marca = time.strftime("%Y%m%d_%H%M%S")
    ruta_salida = os.path.join(CARPETA_REPORTES, f"{caso_id}_resumen_{marca}.pdf")
    pdf.output(ruta_salida)
    return ruta_salida


def abrir_pdf(ruta_pdf):
    """Abre el PDF con el visor por defecto del sistema apenas se
    genera, para que se sienta como una 'descarga automatica' en vez
    de tener que ir a buscar el archivo a la carpeta reportes/. Falla
    en silencio si el sistema operativo no lo soporta (ej. Linux sin
    xdg-open) — quien llama a esta funcion debe seguir mostrando la
    ruta del archivo por si esto no funciona."""
    try:
        if os.name == "nt":
            os.startfile(ruta_pdf)  # Windows
        elif sys_platform_es_mac():
            os.system(f'open "{ruta_pdf}"')
        else:
            os.system(f'xdg-open "{ruta_pdf}"')
    except Exception:
        pass


def sys_platform_es_mac():
    import sys
    return sys.platform == "darwin"
