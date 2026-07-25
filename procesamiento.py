import cv2
import numpy as np


def preparar_y_obtener_bordes(imagen_bgr, mascara_ia):
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    mascara_res = cv2.resize(mascara_ia, (gris.shape[1], gris.shape[0]))
    gris_limpio = cv2.bitwise_and(gris, gris, mask=mascara_res)

    # MEJORA: CLAHE (contraste local adaptativo) antes del blackhat.
    # Una foto de poste suele tener cielo muy brillante y poste/sombra
    # muy oscuros en el mismo cuadro; un contraste global (como hacia
    # Otsu solo) deja cables "lavados" en la zona clara. CLAHE iguala
    # el contraste por regiones para que los cables se vean parejo
    # tanto contra el cielo como contra la fachada de un edificio.
    #
    # AJUSTE: clipLimit mas bajo (2.5 -> 1.2) y bloques mas grandes
    # (8x8 -> 4x4). Con cielos de atardecer (degradado suave naranja/
    # azul), CLAHE con bloques chicos y clip alto exagera esa poca
    # variacion de color hasta inventar bordes falsos por todo el
    # cielo, que el detector de cables termina contando como lineas.
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))
    gris_limpio = clahe.apply(gris_limpio)

    # 1. ESCANER DE CABLES (BlackHat: atrapa lineas oscuras y delgadas)
    # MEJORA: blackhat MULTI-ESCALA. Un cable cercano a la camara ocupa
    # varios pixeles de grosor; uno lejano (o mas fino) puede ser casi
    # un hilo de 1-2px. Un solo kernel (25,25) favorece a un tamano y
    # se come al otro. Usamos dos kernels y fusionamos los resultados.
    cables_mask = np.zeros_like(gris_limpio)
    for tam_kernel in (15, 25, 35):
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (tam_kernel, tam_kernel))
        blackhat = cv2.morphologyEx(gris_limpio, cv2.MORPH_BLACKHAT, kernel_bh)
        _, mascara_escala = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cables_mask = cv2.bitwise_or(cables_mask, mascara_escala)

    # 2. ESCANER DE POSTES (Sobel direccional: masas verticales)
    blur_poste = cv2.GaussianBlur(gris_limpio, (15, 15), 0)
    sobel_x = cv2.Sobel(blur_poste, cv2.CV_64F, 1, 0, ksize=5)
    poste_mask = cv2.convertScaleAbs(sobel_x)
    _, poste_mask = cv2.threshold(poste_mask, 35, 255, cv2.THRESH_BINARY)

    # 3. Fusion y sellado de micro-cortes
    bordes_totales = cv2.bitwise_or(cables_mask, poste_mask)
    bordes_totales = cv2.dilate(bordes_totales, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    return bordes_totales