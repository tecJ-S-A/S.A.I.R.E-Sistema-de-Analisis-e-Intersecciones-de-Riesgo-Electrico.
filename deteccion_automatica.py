"""
deteccion_automatica.py
Reemplazo del disparo por Arduino: usa la webcam y OpenCV para detectar
automaticamente cuando el personaje LEGO SE DETIENE frente al poste, y
en ese momento captura la foto y corre el mismo analisis de S.A.I.R.E.
(el MISMO pipeline que usa main.py -antes estaba copiado y pegado aca,
ahora se importa de pipeline.py para que nunca se desincronicen).

Logica (maquina de estados simple):
  ESPERANDO_MOVIMIENTO -> se detecta movimiento -> EN_MOVIMIENTO
  EN_MOVIMIENTO -> el movimiento baja del umbral durante varios frames seguidos -> DETENIDO
  DETENIDO -> se captura la foto y se ejecuta el analisis -> vuelve a ESPERANDO_MOVIMIENTO

Respaldo: en cualquier momento, presionar la tecla ENTER captura manualmente.
Presionar ESC cierra el programa.

Uso:
    python deteccion_automatica.py
"""

import sys
import os
import cv2

ruta_actual = os.path.dirname(os.path.abspath(__file__))
if ruta_actual not in sys.path:
    sys.path.insert(0, ruta_actual)

from pipeline import ejecutar_pipeline, resumen_texto
from historial import guardar_registro

# --- Parametros ajustables (calibrar ANTES del evento, con la iluminacion real del lugar) ---
UMBRAL_MOVIMIENTO = 2500
FRAMES_QUIETO_PARA_DISPARAR = 15
FRAMES_MOVIMIENTO_MINIMO = 5


def procesar_imagen(frame):
    """Corre el pipeline de S.A.I.R.E. sobre un frame capturado, lo
    guarda en el historial (igual que hace main.py con cada foto) y
    muestra el resultado en una ventana lado a lado."""
    resultado = ejecutar_pipeline(frame)
    texto, hay_alerta = resumen_texto(resultado["postes"])
    print(f"\n>> {texto}\n")
    if hay_alerta:
        print(">> ATENCION: revisar este resultado (alerta o poste no identificado).\n")

    # MEJORA: antes las capturas automaticas por webcam se mostraban en
    # pantalla pero nunca quedaban guardadas en el historial de
    # auditorias (a diferencia de "Cargar Imagen" / "Tomar Foto" en
    # main.py). Se guarda cada poste detectado con el mismo formato que
    # usa la interfaz, para no perder ningun registro del evento.
    for poste in resultado["postes"]:
        try:
            guardar_registro(resultado["img_procesada"], poste["tipo"], poste["cables"], poste["estado"])
        except Exception as exc:  # noqa: BLE001 - no debe tumbar la deteccion en vivo
            print(f">> AVISO: no se pudo guardar en el historial: {exc}")

    dashboard = cv2.hconcat([
        cv2.resize(resultado["img_original"], (640, 480)),
        cv2.resize(resultado["img_procesada"], (640, 480)),
    ])
    cv2.imshow("S.A.I.R.E. - Resultado (presiona cualquier tecla para continuar)", dashboard)
    cv2.waitKey(0)
    cv2.destroyWindow("S.A.I.R.E. - Resultado (presiona cualquier tecla para continuar)")


def main():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: no se pudo abrir la camara web. Revisa que no este siendo usada por otro programa.")
        return

    fondo_previo = None
    estado = "ESPERANDO_MOVIMIENTO"
    contador_quieto = 0
    contador_movimiento = 0

    print("=== S.A.I.R.E. - Disparo automatico por movimiento ===")
    print("Esperando a que el personaje LEGO se mueva y luego se detenga frente al poste...")
    print("(Respaldo manual: presiona ENTER en cualquier momento para capturar. ESC para salir.)\n")

    while True:
        ok, frame = cam.read()
        if not ok:
            print("ERROR: se perdio la senal de la camara.")
            break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (21, 21), 0)

        if fondo_previo is None:
            fondo_previo = gris
            continue

        diferencia = cv2.absdiff(fondo_previo, gris)
        _, mascara = cv2.threshold(diferencia, 25, 255, cv2.THRESH_BINARY)
        nivel_movimiento = cv2.countNonZero(mascara)
        fondo_previo = gris

        cv2.putText(frame, f"Estado: {estado}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Camara S.A.I.R.E. (ESC para salir, ENTER para captura manual)", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:  # ESC
            break
        if tecla == 13:  # ENTER = captura manual de respaldo
            print(">> Captura manual solicitada.")
            procesar_imagen(frame)
            estado, contador_quieto, contador_movimiento = "ESPERANDO_MOVIMIENTO", 0, 0
            continue

        if estado == "ESPERANDO_MOVIMIENTO":
            if nivel_movimiento > UMBRAL_MOVIMIENTO:
                contador_movimiento += 1
                if contador_movimiento >= FRAMES_MOVIMIENTO_MINIMO:
                    estado = "EN_MOVIMIENTO"
                    contador_quieto = 0
            else:
                contador_movimiento = 0

        elif estado == "EN_MOVIMIENTO":
            if nivel_movimiento < UMBRAL_MOVIMIENTO:
                contador_quieto += 1
                if contador_quieto >= FRAMES_QUIETO_PARA_DISPARAR:
                    print(">> Movimiento detenido: capturando y analizando...")
                    procesar_imagen(frame)
                    estado, contador_quieto, contador_movimiento = "ESPERANDO_MOVIMIENTO", 0, 0
            else:
                contador_quieto = 0

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
