"""
Evaluacion de precision por lotes.

Esto es lo que de verdad responde la pregunta "¿funciona bien con
cualquier imagen?" -no una sola foto probada a ojo, sino un conjunto
de fotos con un conteo real conocido, y un numero de error medible.

PASO 1: Juntá entre 10 y 20 fotos reales (mientras mas variadas mejor:
sol, sombra, cielo despejado, fondo de edificios, poste cerca/lejos).
Ponelas todas en una carpeta, por ejemplo "fotos_test/".

PASO 2: Contá los cables de cada foto VOS MISMO, a ojo, con cuidado.
Cargá ese numero en el archivo conteo_real.json (mismo formato que el
ejemplo que te dejo mas abajo, se genera solo la primera vez que corras
este script si no existe).

PASO 3: Correlo:
    python evaluar_precision.py fotos_test/

Te va a imprimir una tabla comparando, para cada foto:
    - tu conteo real
    - lo que da el metodo de picos (el que usa main.py hoy)
    - lo que da el metodo de lineas Hough (el nuevo)
    - el error de cada uno

Y al final, el error promedio (MAE) de cada metodo sobre todo el lote.
El metodo con MENOR error promedio es el que conviene dejar en
main.py / analisis.py como definitivo.
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
from analisis import detectar_estructura, detectar_cables, contar_cables_hough

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png")


def _cargar_conteo_real(carpeta):
    ruta_json = os.path.join(carpeta, "conteo_real.json")
    if not os.path.exists(ruta_json):
        print(f"No encontre {ruta_json}.")
        print("Creando una plantilla vacia para que la completes...")
        fotos = [f for f in os.listdir(carpeta) if f.lower().endswith(EXTENSIONES_VALIDAS)]
        plantilla = {foto: None for foto in fotos}
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(plantilla, f, ensure_ascii=False, indent=2)
        print(f"Completa los numeros en {ruta_json} (reemplaza los 'null' por el conteo real de cada foto) y volve a correr este script.")
        sys.exit(0)

    with open(ruta_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _analizar_foto(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    if img is None:
        return None
    img = cv2.resize(img, (1280, 720))

    mascara, _, _ = analizar_escena(img)
    bordes = preparar_y_obtener_bordes(img, mascara)
    ancho_poste, bbox_poste = detectar_estructura(bordes)

    conteo_picos, _ = detectar_cables(bordes, bbox_poste)
    alto, ancho = bordes.shape[:2]
    conteo_hough, _ = contar_cables_hough(bordes, bbox_poste, alto, ancho)

    return conteo_picos, conteo_hough, bbox_poste is not None, ancho_poste


def evaluar(carpeta):
    conteo_real = _cargar_conteo_real(carpeta)

    filas = []
    for archivo, real in conteo_real.items():
        if real is None:
            print(f"AVISO: '{archivo}' no tiene conteo real cargado en conteo_real.json, se omite.")
            continue

        ruta_imagen = os.path.join(carpeta, archivo)
        resultado = _analizar_foto(ruta_imagen)
        if resultado is None:
            print(f"AVISO: no se pudo abrir '{archivo}', se omite.")
            continue

        conteo_picos, conteo_hough, poste_detectado, ancho_poste = resultado
        filas.append((archivo, real, conteo_picos, conteo_hough, poste_detectado, ancho_poste))

    if not filas:
        print("No hay fotos con conteo real cargado todavia. Completa conteo_real.json.")
        return

    print(f"\n{'Foto':<30} {'Real':>6} {'Picos':>8} {'Hough':>8} {'Err.Picos':>11} {'Err.Hough':>11} {'Poste':>8} {'AnchoPos':>9}")
    print("-" * 100)

    suma_error_picos = 0
    suma_error_hough = 0
    for archivo, real, picos, hough, poste_detectado, ancho_poste in filas:
        err_picos = abs(real - picos)
        err_hough = abs(real - hough)
        suma_error_picos += err_picos
        suma_error_hough += err_hough
        estado_poste = "SI" if poste_detectado else "NO <--"
        print(f"{archivo:<30} {real:>6} {picos:>8} {hough:>8} {err_picos:>11} {err_hough:>11} {estado_poste:>8} {ancho_poste:>9}")

    n = len(filas)
    print("-" * 78)
    print(f"Error promedio (MAE) - metodo de PICOS:  {suma_error_picos / n:.2f} cables")
    print(f"Error promedio (MAE) - metodo de HOUGH:   {suma_error_hough / n:.2f} cables")
    print(f"\nTotal de fotos evaluadas: {n}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python evaluar_precision.py ruta/a/carpeta_con_fotos/")
        sys.exit(1)
    evaluar(sys.argv[1])