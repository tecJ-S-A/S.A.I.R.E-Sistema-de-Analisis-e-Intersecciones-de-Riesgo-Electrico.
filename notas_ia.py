"""
notas_ia.py
Convierte una nota de campo escrita a mano por el usuario (ej: "esta
lloviendo y un cable esta chispeando cerca del poste") en una
observacion redactada de forma mas profesional para el informe.

Funciona en DOS modos, sin que el usuario tenga que configurar nada:

1. MODO OFFLINE (por defecto, siempre disponible): una limpieza de
   texto simple con reglas (capitalizar, puntuar, envolver en una
   frase estandar). No necesita internet ni API key. Es lo que se usa
   en la demo del hackathon si no hay conexion.

2. MODO IA (opcional): si existe la variable de entorno
   ANTHROPIC_API_KEY, se usa el modelo para reformular la nota de
   verdad. Si falla por cualquier motivo (sin internet, sin cuota,
   libreria no instalada), cae automaticamente al modo offline sin
   romper el flujo — el usuario nunca deberia ver un error por esto,
   en el peor caso su nota queda un poco menos pulida.
"""

import os


def pulir_nota(texto_usuario):
    texto_usuario = (texto_usuario or "").strip()
    if not texto_usuario:
        return ""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return _pulir_con_ia(texto_usuario, api_key)
        except Exception:
            pass  # sin internet / sin cuota / libreria faltante -> modo offline

    return _pulir_offline(texto_usuario)


def _pulir_offline(texto):
    texto = texto.strip()
    texto = texto[0].upper() + texto[1:]
    if not texto.endswith((".", "!", "?")):
        texto += "."
    return f"Observacion registrada en campo: {texto}"


def _pulir_con_ia(texto, api_key):
    # Import diferido a proposito: si la libreria "anthropic" no esta
    # instalada, el resto de la app sigue funcionando igual (solo se
    # usa el modo offline). No hace falta agregarla a requirements.txt
    # si no van a usar esta funcion.
    from anthropic import Anthropic

    cliente = Anthropic(api_key=api_key)
    respuesta = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": (
                "Reescribe esta nota de campo de una auditoria de postes "
                "electricos como una observacion tecnica breve y profesional, "
                "en espanol, en 1 o 2 frases, sin inventar datos que no esten "
                "en la nota original:\n\n" + texto
            ),
        }],
    )
    return respuesta.content[0].text.strip()
