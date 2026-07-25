"""
modelo_ml.py
Puente entre el pipeline en vivo y el modelo entrenado por
entrenar_modelo_cables.py (si existe). Disenado para que su ausencia
NUNCA rompa la app:

  - Si modelo_cables.pkl no existe (todavia no entrenaron nada): se
    devuelve None y pipeline.py sigue usando el metodo heuristico.
  - Si scikit-learn/joblib no estan instalados: idem, None y sigue el
    heuristico.
  - Si el modelo existe pero falla al predecir por cualquier motivo:
    idem, None.

Asi, el modelo de ML es una MEJORA opcional que se activa sola en
cuanto corren entrenar_modelo_cables.py, sin que nadie tenga que tocar
main.py ni pipeline.py para prenderla o apagarla.
"""

import os

_modelo = None
_intentado_cargar = False


def _cargar_modelo():
    global _modelo, _intentado_cargar
    if _intentado_cargar:
        return _modelo
    _intentado_cargar = True

    ruta_modelo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_cables.pkl")
    if not os.path.exists(ruta_modelo):
        return None

    try:
        import joblib
        _modelo = joblib.load(ruta_modelo)
    except Exception:
        _modelo = None
    return _modelo


def predecir_estimado_ml(ancho_poste, picos_izq, picos_der, densidad):
    """Devuelve un entero (el punto central estimado de cables) si hay
    un modelo entrenado disponible, o None si no lo hay / algo fallo.
    pipeline.py sigue calculando el RANGO alrededor de este punto con
    la misma logica de siempre (estimar_rango_cables) -- lo unico que
    cambia es de donde sale el centro del rango."""
    modelo = _cargar_modelo()
    if modelo is None:
        return None

    features = [[
        ancho_poste, picos_izq, picos_der,
        abs(picos_izq - picos_der), densidad,
        1 if ancho_poste else 0,
    ]]
    try:
        prediccion = modelo.predict(features)[0]
    except Exception:
        return None

    return max(0, round(prediccion))
