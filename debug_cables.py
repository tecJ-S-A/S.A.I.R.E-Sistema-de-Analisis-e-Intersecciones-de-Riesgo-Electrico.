"""
Script de calibracion visual para el conteo de cables.

Uso:
    python debug_cables.py ruta/a/tu/foto.jpg

Que hace:
    1. Corre el mismo pipeline que main.py (YOLO -> mascara -> bordes ->
       deteccion de poste) sobre UNA foto.
    2. Dibuja sobre la imagen las dos franjas verticales que se usan
       para contar cables (izquierda y derecha del poste).
    3. Grafica el perfil de cada franja (cuantos pixeles blancos hay
       en cada fila) y marca con una 'x' cada pico detectado, es
       decir, cada cable que la funcion cuenta.

Sirve para responder a ojo: "¿el pico que se detecto ahi realmente es
un cable, o es ruido? ¿le falto detectar alguno?" y ajustar en
consecuencia los parametros de analisis.py:
    - ancho_franja      (que tan ancha es la banda que se analiza)
    - margen_poste      (que tan lejos del poste se ubica la banda)
    - distance en find_peaks   (separacion minima entre dos cables)
    - height en find_peaks     (que tan "lleno" debe estar un pico)
"""
import sys
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # cambia a "TkAgg" si corres esto localmente y queres plt.show() interactivo
import matplotlib.pyplot as plt

ruta_actual = os.path.dirname(os.path.abspath(__file__))
if ruta_actual not in sys.path:
    sys.path.insert(0, ruta_actual)

from vision_ia import analizar_escena
from procesamiento import preparar_y_obtener_bordes
from analisis import detectar_estructura, _contar_picos_en_franja

# Deben coincidir con los valores usados en detectar_cables() (analisis.py)
MARGEN_POSTE = 20
ANCHO_FRANJA = 18


def analizar_una_foto(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"No se pudo abrir la imagen: {ruta_imagen}")
        return
    img = cv2.resize(img, (1280, 720))

    mascara, _, _ = analizar_escena(img)
    bordes = preparar_y_obtener_bordes(img, mascara)
    ancho_poste, bbox_poste = detectar_estructura(bordes)

    if not bbox_poste:
        print("No se detecto un poste en esta foto: no se puede ubicar la franja de conteo.")
        _mostrar_solo_bordes(img, bordes, ruta_imagen)
        return

    alto, ancho = bordes.shape[:2]
    x1, _, x2, _ = bbox_poste

    franja_izq_fin = max(0, x1 - MARGEN_POSTE)
    franja_izq_ini = max(0, franja_izq_fin - ANCHO_FRANJA)
    franja_der_ini = min(ancho, x2 + MARGEN_POSTE)
    franja_der_fin = min(ancho, franja_der_ini + ANCHO_FRANJA)

    picos_izq, perfil_izq, indices_izq = _contar_picos_en_franja(bordes, franja_izq_ini, franja_izq_fin, alto)
    picos_der, perfil_der, indices_der = _contar_picos_en_franja(bordes, franja_der_ini, franja_der_fin, alto)

    print(f"Poste detectado: ancho={ancho_poste}px, bbox={bbox_poste}")
    print(f"Franja izquierda -> {picos_izq} cables detectados")
    print(f"Franja derecha   -> {picos_der} cables detectados")
    print(f"Conteo final (max de ambas) = {max(picos_izq, picos_der)}")

    _graficar(img, bordes, bbox_poste,
              (franja_izq_ini, franja_izq_fin), (franja_der_ini, franja_der_fin),
              perfil_izq, perfil_der, indices_izq, indices_der, ruta_imagen)


def _graficar(img, bordes, bbox_poste, franja_izq, franja_der,
              perfil_izq, perfil_der, indices_izq, indices_der, ruta_imagen):
    x1p, _, x2p, _ = bbox_poste

    img_marcada = img.copy()
    cv2.rectangle(img_marcada, (x1p, 0), (x2p, img.shape[0]), (0, 255, 255), 2)
    cv2.rectangle(img_marcada, (franja_izq[0], 0), (franja_izq[1], img.shape[0]), (255, 0, 0), 2)
    cv2.rectangle(img_marcada, (franja_der[0], 0), (franja_der[1], img.shape[0]), (0, 0, 255), 2)
    img_rgb = cv2.cvtColor(img_marcada, cv2.COLOR_BGR2RGB)

    fig, ejes = plt.subplots(1, 3, figsize=(18, 6))

    ejes[0].imshow(img_rgb)
    ejes[0].set_title("Poste (amarillo) y franjas de conteo\n(azul=izq, rojo=der)")
    ejes[0].axis("off")

    for eje, perfil, indices_picos, color, nombre in (
        (ejes[1], perfil_izq, indices_izq, "tab:blue", "Franja izquierda"),
        (ejes[2], perfil_der, indices_der, "tab:red", "Franja derecha"),
    ):
        if perfil is None:
            eje.set_title(f"{nombre}: sin datos")
            continue
        y = np.arange(len(perfil))
        eje.plot(perfil, y, color=color)
        if len(indices_picos) > 0:
            eje.plot(perfil[indices_picos], indices_picos, "kx", markersize=10, markeredgewidth=2)
        eje.invert_yaxis()  # para que arriba de la imagen quede arriba del grafico
        eje.set_title(f"{nombre}: {len(indices_picos)} picos = {len(indices_picos)} cables")
        eje.set_xlabel("pixeles blancos en esa fila")
        eje.set_ylabel("posicion vertical (y)")

    plt.tight_layout()
    nombre_salida = os.path.splitext(os.path.basename(ruta_imagen))[0]
    ruta_salida = os.path.join(ruta_actual, f"debug_{nombre_salida}.png")
    plt.savefig(ruta_salida, dpi=120)
    print(f"\nGrafico guardado en: {ruta_salida}")


def _mostrar_solo_bordes(img, bordes, ruta_imagen):
    fig, ejes = plt.subplots(1, 2, figsize=(14, 6))
    ejes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ejes[0].set_title("Original")
    ejes[0].axis("off")
    ejes[1].imshow(bordes, cmap="gray")
    ejes[1].set_title("Bordes detectados (sin poste)")
    ejes[1].axis("off")
    plt.tight_layout()
    nombre_salida = os.path.splitext(os.path.basename(ruta_imagen))[0]
    ruta_salida = os.path.join(ruta_actual, f"debug_{nombre_salida}.png")
    plt.savefig(ruta_salida, dpi=120)
    print(f"Grafico guardado en: {ruta_salida}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python debug_cables.py ruta/a/tu/foto.jpg")
        sys.exit(1)
    analizar_una_foto(sys.argv[1])