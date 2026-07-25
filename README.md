# S.A.I.R.E. PRO
Sistema de Auditoria Inteligente para Riesgo Electrico.

Analiza fotos (o video en vivo) de postes de luz, detecta el poste y los
cables con vision por computadora, ignora ruido urbano y vegetacion con
IA, y evalua el riesgo de congestion de cables para auditoria municipal.

## Instalacion

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

pip install -r requirements.txt
python main.py
```

La primera vez que se ejecuta, Ultralytics descarga automaticamente los
pesos de **YOLO26n** (`yolo26n.pt`) si no estan ya en la carpeta del
proyecto y hay conexion a internet. Si no hay conexion, la aplicacion
cae automaticamente a `yolov8n.pt` (incluido en el proyecto) para no
dejar de funcionar.

## Estructura

| Archivo | Rol |
|---|---|
| `main.py` | Interfaz (Tkinter): captura de imagen/camara, orquestacion y dashboard |
| `vision_ia.py` | Carga del modelo YOLO y filtrado de ruido urbano + vegetacion |
| `procesamiento.py` | Extraccion de bordes (blackhat para cables, Sobel para postes) |
| `analisis.py` | Deteccion de estructura del poste, conteo de cables, evaluacion de riesgo |
| `historial.py` | Persistencia del historial de auditorias |
| `debug_cables.py` | Herramienta de calibracion visual del conteo de cables |
| `evaluar_precision.py` | Evaluacion de precision por lotes (MAE) |

## Cambios de esta revision

### 1. Migracion a YOLO26
`vision_ia.py` ahora carga `yolo26n.pt` (release de Ultralytics de enero
2026: inferencia end-to-end sin NMS, mejor precision y hasta ~43% mas
rapido en CPU que YOLO11n) en vez de `yolov8n.pt`. La interfaz de
Python es identica a la de YOLOv8, y ambos se entrenan sobre COCO, asi
que los IDs de clase que se ignoran (persona, auto, etc.) no cambiaron.
Si YOLO26 no esta disponible en el equipo, el programa cae automatica
y silenciosamente a YOLOv8n en vez de fallar.

### 2. Bugs corregidos

- **Ruta del modelo relativa** (`vision_ia.py`): se cargaba con
  `YOLO("yolov8n.pt")`, una ruta relativa al directorio de trabajo. Si
  el programa se lanzaba desde otra carpeta, no encontraba el archivo
  local. Ahora se resuelve una ruta absoluta segun la ubicacion real
  del script.
- **Crash total al fallar la carga del modelo**: si el archivo de
  pesos faltaba, la app se caia con una traza de consola antes de
  mostrar cualquier ventana. Ahora hay manejo de errores con fallback
  y un mensaje claro para el usuario.
- **Interfaz congelada durante el analisis** (`main.py`): el pipeline
  completo (YOLO + procesamiento de imagen) corria en el hilo
  principal de Tkinter, asi que la ventana quedaba en "No responde"
  cada vez que se analizaba una foto. Ahora corre en un hilo aparte
  con una barra de progreso, sin bloquear la interfaz.
- **Sin verificacion de camara**: `tomar_foto()` abria la ventana de
  captura sin comprobar si la camara realmente se pudo abrir; si no
  habia camara, quedaba un cuadro vacio para siempre sin explicacion.
  Ahora se detecta y se avisa, tanto al abrir como si la senal se
  pierde a mitad de la captura.
- **Carga de imagen invalida sin aviso**: si `cv2.imread()` fallaba
  (archivo corrupto, ruta invalida), el boton "Cargar Imagen"
  simplemente no hacia nada visible. Ahora se informa el error.
- **Miniaturas huerfanas en disco** (`historial.py`): el historial se
  recortaba a 50 registros en el JSON, pero los archivos de imagen de
  los registros descartados nunca se borraban del disco: crecian para
  siempre. Ahora se eliminan junto con el registro.
- **Colision de nombres de archivo** (`historial.py`): dos capturas en
  el mismo segundo generaban el mismo nombre de archivo de miniatura y
  una sobrescribia a la otra en silencio. Se agrego un sufijo unico.
- **Crash de la ventana de Historial**: si una miniatura referenciada
  en el JSON ya no existia o estaba corrupta, `Image.open()` lanzaba
  una excepcion sin atrapar que tumbaba toda la ventana de historial.
  Ahora ese registro puntual se muestra sin imagen, sin afectar al
  resto.
- **Vegetacion indistinguible del ruido urbano**: las cajas de
  vegetacion se dibujaban con el mismo color que las de YOLO (autos,
  personas, etc.), sin etiqueta. Ahora tienen color y etiqueta propios.
- **Sin manera de borrar el historial**: no existia ninguna funcion
  para eliminar un registro o vaciar el historial desde la interfaz.
  Se agrego, con confirmacion antes de vaciar (accion irreversible).
- **Analisis concurrente sin proteccion**: nada impedia lanzar dos
  analisis a la vez (por ejemplo, tocando "Cargar Imagen" varias veces
  seguidas), lo que podia mezclar resultados en la interfaz. Ahora se
  bloquea mientras hay un analisis en curso.
- **Dependencias no documentadas**: no existia `requirements.txt`, asi
  que instalar el proyecto en otra maquina dependia de adivinar que
  librerias hacian falta.

### 3. Mejoras de interfaz

- Ventana redimensionable (antes tenia tamano fijo).
- Analisis en segundo plano con barra de progreso indeterminada.
- Control deslizante de "Sensibilidad IA" (umbral de confianza de
  YOLO) ajustable en vivo, en vez de un valor fijo en el codigo.
- Boton "Exportar Resultado" para guardar la imagen procesada a
  resolucion completa donde el usuario elija (antes solo quedaba la
  miniatura chica del historial).
- Barra de menu (Archivo / Ver / Ayuda) con las mismas acciones y un
  cuadro "Acerca de" que muestra el motor de IA activo.
- Historial con boton de borrado por registro y "Vaciar historial".
- Barra de estado con mensajes de error visibles, en vez de fallos
  silenciosos.

## Herramientas de calibracion

```bash
python debug_cables.py ruta/a/tu/foto.jpg      # visualiza el conteo de cables
python evaluar_precision.py postes_test/       # mide el error (MAE) sobre un lote de fotos
```
