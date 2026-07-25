"""
entrenar_modelo_cables.py

Script de ENTRENAMIENTO -- NO es parte de la app en vivo (main.py no lo
importa ni lo necesita para arrancar). Se corre una sola vez, o cada
vez que agreguen mas fotos etiquetadas, para producir modelo_cables.pkl.
Si ese archivo existe, pipeline.py lo usa automaticamente para afinar
el punto central del rango de cables; si no existe (o si scikit-learn
no esta instalado), la app sigue funcionando igual con el metodo
heuristico de siempre. Cero riesgo de romper nada.

POR QUE ESTO Y NO UN DETECTOR YOLO CUSTOM DESDE YA: con 20-40 fotos
etiquetadas, un Random Forest sobre las features que el pipeline YA
calcula (ancho del poste, picos por lado, densidad) puede aprender
patrones utiles sin necesitar cientos de imagenes, GPU, ni horas de
entrenamiento. Es una mejora real y alcanzable en el tiempo que les
queda. Un detector YOLO entrenado a medida (con Roboflow para etiquetar
cajas de "poste" y "manojo_cables") es el siguiente escalon logico
DESPUES del hackathon, cuando tengan mas fotos y mas tiempo.

USO:
    1. pip install scikit-learn joblib
    2. Pon tus fotos en una carpeta, ej: postes_test/
    3. Corre:  python entrenar_modelo_cables.py postes_test/
       La primera vez crea rango_real.json con todos los "null".
    4. Completa el RANGO REAL de cada foto (no el numero exacto -- en
       nudos densos ni una persona puede contarlos uno por uno):
           {
             "poste1.jpg": [8, 10],
             "nudo_mercado.jpg": [25, 35]
           }
       Mismo espiritu que conteo_real.json de evaluar_precision.py,
       pero con un rango en vez de un numero unico.
    5. Vuelve a correr el mismo comando. Esta vez entrena y guarda
       modelo_cables.pkl en la carpeta del proyecto (junto a main.py).
"""
import sys
import os
import json

import cv2

ruta_actual = os.path.dirname(os.path.abspath(__file__))
if ruta_actual not in sys.path:
    sys.path.insert(0, ruta_actual)

from vision_ia import analizar_escena
from procesamiento import preparar_y_obtener_bordes
from analisis import detectar_estructura, detectar_cables_detallado, calcular_densidad_cables

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png")
NOMBRES_FEATURES = ["ancho_poste", "picos_izq", "picos_der", "diferencia_lados", "densidad", "poste_detectado"]


def _extraer_features(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    if img is None:
        return None
    img = cv2.resize(img, (1280, 720))

    mascara, _, _ = analizar_escena(img)
    bordes = preparar_y_obtener_bordes(img, mascara)
    ancho_poste, bbox_poste = detectar_estructura(bordes)
    _, _, picos_izq, picos_der = detectar_cables_detallado(bordes, bbox_poste)
    densidad = calcular_densidad_cables(bordes, bbox_poste)

    return [
        ancho_poste,
        picos_izq,
        picos_der,
        abs(picos_izq - picos_der),
        densidad,
        1 if bbox_poste else 0,
    ]


def _cargar_rango_real(carpeta):
    ruta_json = os.path.join(carpeta, "rango_real.json")
    if not os.path.exists(ruta_json):
        fotos = [f for f in os.listdir(carpeta) if f.lower().endswith(EXTENSIONES_VALIDAS)]
        plantilla = {foto: None for foto in fotos}
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(plantilla, f, ensure_ascii=False, indent=2)
        print(f"Cree {ruta_json} con {len(fotos)} fotos.")
        print("Completa el rango real [minimo, maximo] de cada una (reemplaza los 'null')")
        print("y vuelve a correr este mismo comando.")
        sys.exit(0)
    with open(ruta_json, "r", encoding="utf-8") as f:
        return json.load(f)


def entrenar(carpeta):
    try:
        from sklearn.ensemble import RandomForestRegressor
        import joblib
    except ImportError:
        print("Falta instalar las librerias de entrenamiento. Corre:")
        print("    pip install scikit-learn joblib")
        sys.exit(1)

    rangos_reales = _cargar_rango_real(carpeta)

    X, y = [], []
    for archivo, rango in rangos_reales.items():
        if rango is None:
            print(f"AVISO: '{archivo}' sin rango real cargado todavia, se omite.")
            continue

        ruta = os.path.join(carpeta, archivo)
        features = _extraer_features(ruta)
        if features is None:
            print(f"AVISO: no se pudo abrir '{archivo}', se omite.")
            continue

        punto_medio_real = (rango[0] + rango[1]) / 2
        X.append(features)
        y.append(punto_medio_real)

    print(f"\n{len(X)} fotos con rango real cargado y features extraidas.")

    if len(X) < 8:
        print("Se necesitan al menos ~8-10 fotos etiquetadas para que el modelo")
        print("aprenda algo minimamente util. Sigue etiquetando rango_real.json")
        print("y vuelve a correr este mismo comando cuando tengas mas.")
        return

    modelo = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    modelo.fit(X, y)

    ruta_modelo = os.path.join(ruta_actual, "modelo_cables.pkl")
    joblib.dump(modelo, ruta_modelo)

    print(f"\nModelo entrenado con {len(X)} fotos y guardado en:\n  {ruta_modelo}")
    print("\nImportancia de cada feature en la prediccion (mas alto = mas peso):")
    for nombre, importancia in zip(NOMBRES_FEATURES, modelo.feature_importances_):
        print(f"  {nombre}: {importancia:.2f}")
    print("\npipeline.py ya lo va a usar automaticamente la proxima vez que corran main.py.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python entrenar_modelo_cables.py ruta/a/carpeta_con_fotos/")
        sys.exit(1)
    entrenar(sys.argv[1])
