import sys
import os
import time
import queue
import threading
import urllib.parse
import webbrowser

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk

ruta_actual = os.path.dirname(os.path.abspath(__file__))
if ruta_actual not in sys.path:
    sys.path.insert(0, ruta_actual)

try:
    from vision_ia import NOMBRE_MODELO_ACTIVO, CONF_POR_DEFECTO
    from pipeline import ejecutar_pipeline
    from historial import (
        guardar_registro, cargar_historial, eliminar_registro,
        vaciar_historial, resumen_historial,
    )
    from notas_ia import pulir_nota
    from validacion import buscar_caso_verificado, texto_comparacion
except Exception as _err:
    _root_error = tk.Tk()
    _root_error.withdraw()
    messagebox.showerror(
        "S.A.I.R.E. PRO - Error de inicio",
        "No se pudo inicializar el sistema de analisis.\n\n"
        f"Detalle: {_err}\n\n"
        "Verifica que las dependencias esten instaladas "
        "(pip install -r requirements.txt) y que el archivo de pesos "
        "YOLO (yolo26n.pt o yolov8n.pt) este en la carpeta del proyecto.",
    )
    _root_error.destroy()
    sys.exit(1)

# --- Paleta de colores de la interfaz ---
BG = "#0d1420"
PANEL = "#182430"
PANEL_ALT = "#1f2e3d"
BORDE = "#2b3b4d"
ACENTO = "#f2a900"
ACENTO_HOVER = "#d69400"
TEXTO = "#e8edf2"
TEXTO_SEC = "#93a3b3"
VERDE = "#33cc66"
NARANJA = "#ff9f2e"
ROJO = "#ff4d4f"


class AppSaire:
    def __init__(self, root):
        self.root = root
        self.root.title("S.A.I.R.E. PRO - Auditoria Municipal")
        self.root.configure(bg=BG)
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)

        # Estado interno
        self._cola = queue.Queue()
        self._procesando = False
        self._ultima_imagen_procesada = None
        self._ultima_ruta_imagen_completa = None
        self._ultima_ruta_imagen_original = None
        self._ultimo_registro = None
        self._nombre_archivo_actual = None
        self._miniaturas_historial = []
        self.zona_actual = ""

        self._configurar_estilos_ttk()
        self._construir_menu()
        self._construir_interfaz()
        self._actualizar_estado_inicial()

        # NUEVO: se pregunta la zona apenas arranca la app, asi cada
        # foto que se tome ya queda asociada a un lugar sin que el
        # usuario tenga que escribirlo cada vez. after(300, ...) para
        # que la ventana principal ya este dibujada antes de mostrar
        # el dialogo modal.
        self.root.after(300, self._pedir_zona_inicial)

    # ---------- ESTILOS ----------
    def _configurar_estilos_ttk(self):
        estilo = ttk.Style()
        try:
            estilo.theme_use("clam")
        except tk.TclError:
            pass
        estilo.configure("Saire.Horizontal.TProgressbar", troughcolor=PANEL,
                          background=ACENTO, bordercolor=PANEL, lightcolor=ACENTO,
                          darkcolor=ACENTO)
        estilo.configure("Saire.Horizontal.TScale", background=BG)

    # ---------- MENU ----------
    def _construir_menu(self):
        barra_menu = tk.Menu(self.root)

        menu_archivo = tk.Menu(barra_menu, tearoff=0)
        menu_archivo.add_command(label="Cargar Imagen...", command=self.cargar_imagen)
        menu_archivo.add_command(label="Tomar Foto...", command=self.tomar_foto)
        menu_archivo.add_command(label="Exportar Resultado...", command=self.exportar_resultado)
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Cambiar Zona...", command=self._pedir_zona_inicial)
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Salir", command=self.root.quit)
        barra_menu.add_cascade(label="Archivo", menu=menu_archivo)

        menu_ver = tk.Menu(barra_menu, tearoff=0)
        menu_ver.add_command(label="Historial de Auditorias", command=self.abrir_historial)
        menu_ver.add_command(label="Resumen de Auditorias", command=self._mostrar_resumen)
        menu_ver.add_command(label="Generar Informe PDF (resumen)", command=self._generar_pdf_resumen)
        barra_menu.add_cascade(label="Ver", menu=menu_ver)

        menu_ayuda = tk.Menu(barra_menu, tearoff=0)
        menu_ayuda.add_command(label="Acerca de S.A.I.R.E. PRO", command=self._mostrar_acerca_de)
        barra_menu.add_cascade(label="Ayuda", menu=menu_ayuda)

        self.root.config(menu=barra_menu)

    def _mostrar_acerca_de(self):
        messagebox.showinfo(
            "Acerca de S.A.I.R.E. PRO",
            "S.A.I.R.E. PRO\n"
            "Sistema de Auditoria Inteligente para Riesgo Electrico\n\n"
            f"Motor de vision: {NOMBRE_MODELO_ACTIVO}\n"
            "Deteccion de postes y cables por vision por computadora\n"
            "(YOLO + OpenCV) para evaluar riesgo por congestion de cables.",
        )

    # ---------- INTERFAZ ----------
    def _construir_interfaz(self):
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # --- Encabezado ---
        encabezado = tk.Frame(self.root, bg=BG)
        encabezado.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 0))
        encabezado.grid_columnconfigure(0, weight=1)

        bloque_titulo = tk.Frame(encabezado, bg=BG)
        bloque_titulo.grid(row=0, column=0, sticky="w")
        tk.Label(bloque_titulo, text="S.A.I.R.E. PRO", bg=BG, fg=ACENTO,
                  font=("Segoe UI", 22, "bold")).pack(anchor="w")
        tk.Label(bloque_titulo, text="Auditoria Inteligente de Postes y Cables", bg=BG, fg=TEXTO,
                  font=("Segoe UI", 11)).pack(anchor="w")

        self.lbl_modelo = tk.Label(
            encabezado, text=f"Motor de IA: {NOMBRE_MODELO_ACTIVO}",
            bg=BG, fg=TEXTO_SEC, font=("Segoe UI", 9, "bold"))
        self.lbl_modelo.grid(row=0, column=1, sticky="e")

        # --- Barra de herramientas ---
        barra = tk.Frame(self.root, bg=BG)
        barra.grid(row=1, column=0, sticky="ew", padx=20, pady=12)

        self.btn_cargar = self._boton(barra, "Cargar Imagen", self.cargar_imagen)
        self.btn_cargar.pack(side="left", padx=(0, 6))
        self.btn_camara = self._boton(barra, "Tomar Foto", self.tomar_foto)
        self.btn_camara.pack(side="left", padx=6)
        self.btn_historial = self._boton(barra, "Historial", self.abrir_historial)
        self.btn_historial.pack(side="left", padx=6)
        self.btn_exportar = self._boton(barra, "Exportar Resultado", self.exportar_resultado)
        self.btn_exportar.pack(side="left", padx=6)
        self.btn_exportar.configure(state="disabled")
        self.btn_pdf = self._boton(barra, "Generar Informe PDF", self._generar_pdf_individual)
        self.btn_pdf.pack(side="left", padx=6)
        self.btn_pdf.configure(state="disabled")

        bloque_sensibilidad = tk.Frame(barra, bg=BG)
        bloque_sensibilidad.pack(side="right")
        tk.Label(bloque_sensibilidad, text="Sensibilidad IA:", bg=BG, fg=TEXTO_SEC,
                  font=("Segoe UI", 9)).pack(side="left", padx=(0, 6))
        self.conf_var = tk.DoubleVar(value=CONF_POR_DEFECTO)
        self.lbl_sensibilidad = tk.Label(bloque_sensibilidad, text=f"{CONF_POR_DEFECTO:.2f}",
                                          bg=BG, fg=TEXTO, font=("Segoe UI", 9, "bold"), width=4)
        self.lbl_sensibilidad.pack(side="right", padx=(6, 0))
        control_sensibilidad = ttk.Scale(
            bloque_sensibilidad, from_=0.10, to=0.80, orient="horizontal", length=140,
            variable=self.conf_var, style="Saire.Horizontal.TScale",
            command=self._on_cambio_sensibilidad)
        control_sensibilidad.pack(side="right")

        # --- Panel de imagenes ---
        panel_imgs = tk.Frame(self.root, bg=BG)
        panel_imgs.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 8))
        panel_imgs.grid_columnconfigure(0, weight=1)
        panel_imgs.grid_columnconfigure(1, weight=1)
        panel_imgs.grid_rowconfigure(1, weight=1)

        tk.Label(panel_imgs, text="ORIGINAL", bg=BG, fg=TEXTO_SEC,
                  font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(panel_imgs, text="PROCESADA (deteccion IA)", bg=BG, fg=TEXTO_SEC,
                  font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="w", pady=(0, 4))

        self.lbl_original = tk.Label(panel_imgs, bg=PANEL, text="Esperando una imagen...",
                                      fg=TEXTO_SEC, highlightbackground=BORDE, highlightthickness=1)
        self.lbl_original.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.lbl_procesada = tk.Label(panel_imgs, bg=PANEL, text="Aqui se vera el resultado",
                                       fg=TEXTO_SEC, highlightbackground=BORDE, highlightthickness=1)
        self.lbl_procesada.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

        # --- Barra de estado ---
        barra_estado = tk.Frame(self.root, bg=PANEL_ALT)
        barra_estado.grid(row=3, column=0, sticky="ew")
        barra_estado.grid_columnconfigure(0, weight=1)

        self.lbl_estado = tk.Label(barra_estado, text="Esperando una imagen...", bg=PANEL_ALT,
                                    fg=TEXTO, font=("Segoe UI", 12, "bold"), anchor="w")
        self.lbl_estado.grid(row=0, column=0, sticky="ew", padx=20, pady=10)

        self.progreso = ttk.Progressbar(barra_estado, mode="indeterminate", length=160,
                                         style="Saire.Horizontal.TProgressbar")
        self.progreso.grid(row=0, column=1, padx=20, pady=10)
        self.progreso.grid_remove()

    def _boton(self, padre, texto, comando):
        return tk.Button(padre, text=texto, command=comando, bg=ACENTO, fg="#101820",
                          font=("Segoe UI", 10, "bold"), relief="flat", padx=14, pady=6,
                          activebackground=ACENTO_HOVER, cursor="hand2")

    def _on_cambio_sensibilidad(self, _valor):
        self.lbl_sensibilidad.configure(text=f"{self.conf_var.get():.2f}")

    def _actualizar_estado_inicial(self):
        self.lbl_estado.configure(text="Esperando una imagen...", fg=TEXTO)

    # ---------- ZONA (reemplazo liviano de GPS real) ----------
    def _pedir_zona_inicial(self):
        # NOTA: esta app corre en una laptop, no en un celular, asi que
        # no hay una API de geolocalizacion nativa confiable como en un
        # navegador o app movil. En vez de inventar coordenadas falsas,
        # se pide la zona/direccion aproximada UNA VEZ y se reutiliza en
        # cada registro — suficiente para armar el "mapa de zonas de
        # riesgo" que se muestra en el resumen y en el PDF agregado. Si
        # mas adelante corren esto desde un celular con GPS real, aca es
        # donde reemplazarian el simpledialog por la lectura de
        # coordenadas reales.
        zona = simpledialog.askstring(
            "Zona de la auditoria",
            "¿En que zona, distrito o direccion aproximada se van a\n"
            "tomar las fotos de esta ronda? (Se guarda con cada registro)",
            parent=self.root,
        )
        self.zona_actual = zona.strip() if zona else self.zona_actual
        titulo = "S.A.I.R.E. PRO - Auditoria Municipal"
        if self.zona_actual:
            titulo += f" | Zona: {self.zona_actual}"
        self.root.title(titulo)

    # ---------- ENTRADA DE IMAGEN ----------
    def cargar_imagen(self):
        if self._procesando:
            messagebox.showinfo("S.A.I.R.E. PRO", "Ya hay un analisis en curso, espera a que termine.")
            return
        ruta = filedialog.askopenfilename(title="Cargar Foto de Auditoria",
                                           filetypes=[("Imagenes", "*.jpg *.jpeg *.png")])
        if not ruta:
            return
        img = cv2.imread(ruta)
        if img is None:
            self._mostrar_error(f"No se pudo abrir la imagen:\n{os.path.basename(ruta)}")
            return
        self._nombre_archivo_actual = os.path.basename(ruta)
        self._iniciar_analisis(img)

    def tomar_foto(self):
        if self._procesando:
            messagebox.showinfo("S.A.I.R.E. PRO", "Ya hay un analisis en curso, espera a que termine.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            self._mostrar_error("No se detecto ninguna camara disponible.")
            return

        ventana = tk.Toplevel(self.root)
        ventana.title("Tomar Foto")
        ventana.configure(bg=BG)
        lbl_video = tk.Label(ventana, bg=PANEL)
        lbl_video.pack(padx=10, pady=10)
        lbl_aviso = tk.Label(ventana, text="", bg=BG, fg=ROJO, font=("Segoe UI", 9))
        lbl_aviso.pack()
        estado_captura = {"frame": None, "activo": True}

        def refrescar():
            if not estado_captura["activo"]:
                return
            ok, frame = cap.read()
            if ok:
                estado_captura["frame"] = frame
                lbl_video.imgtk = self._cv2_a_imagetk(frame, (480, 360))
                lbl_video.configure(image=lbl_video.imgtk)
                ventana.after(30, refrescar)
            else:
                lbl_aviso.configure(text="Se perdio la senal de la camara.")
                estado_captura["activo"] = False

        def capturar():
            estado_captura["activo"] = False
            cap.release()
            ventana.destroy()
            if estado_captura["frame"] is not None:
                self._nombre_archivo_actual = None  # foto de camara, no es del banco verificado
                self._iniciar_analisis(estado_captura["frame"])

        def cerrar():
            estado_captura["activo"] = False
            cap.release()
            ventana.destroy()

        self._boton(ventana, "Capturar", capturar).pack(pady=(0, 10))
        ventana.protocol("WM_DELETE_WINDOW", cerrar)
        refrescar()

    # ---------- PIPELINE DE ANALISIS (en segundo plano) ----------
    def _iniciar_analisis(self, img_bruta):
        self._procesando = True
        self._set_botones_ocupados(True)
        self.lbl_estado.configure(text="Analizando imagen...", fg=ACENTO)
        self.progreso.grid()
        self.progreso.start(12)

        conf_actual = self.conf_var.get()
        hilo = threading.Thread(target=self._trabajo_analisis, args=(img_bruta, conf_actual), daemon=True)
        hilo.start()
        self.root.after(80, self._revisar_cola)

    def _trabajo_analisis(self, img_bruta, conf):
        """Corre fuera del hilo principal: NO debe tocar widgets de Tk."""
        try:
            resultado = ejecutar_pipeline(img_bruta, conf=conf)
            poste = resultado["postes"][0]
            self._cola.put(("ok", (resultado["img_original"], resultado["img_procesada"], poste)))
        except Exception as exc:
            self._cola.put(("error", str(exc)))

    def _revisar_cola(self):
        try:
            tipo_msg, payload = self._cola.get_nowait()
        except queue.Empty:
            if self._procesando:
                self.root.after(80, self._revisar_cola)
            return

        self.progreso.stop()
        self.progreso.grid_remove()
        self._procesando = False
        self._set_botones_ocupados(False)

        if tipo_msg == "ok":
            img_original, img_procesada, poste = payload
            self._ultima_imagen_procesada = img_procesada
            self._mostrar_imagenes(img_original, img_procesada)
            self._actualizar_estado(poste)
            self.btn_exportar.configure(state="normal")
            self.btn_pdf.configure(state="normal")
            self._ultima_ruta_imagen_completa = self._guardar_temp_imagen_completa(
                img_procesada, "ultimo_resultado.jpg")
            self._ultima_ruta_imagen_original = self._guardar_temp_imagen_completa(
                img_original, "ultimo_original.jpg")

            # NUEVO: nota de campo opcional, se pulsa una sola vez por
            # foto justo despues del analisis (no interrumpe el hilo de
            # analisis porque ya termino, esto corre en el hilo
            # principal de Tk).
            notas_usuario = self._pedir_nota_campo()
            notas_pulidas = pulir_nota(notas_usuario) if notas_usuario else ""

            try:
                guardar_registro(
                    img_procesada, poste["tipo"], poste["cables_min"], poste["cables_max"],
                    poste["estado"], recomendacion=poste["recomendacion"], zona=self.zona_actual,
                    notas_usuario=notas_usuario, notas_pulidas=notas_pulidas,
                    nudo_detectado=poste["nudo_detectado"],
                    num_nudos_aprox=poste.get("num_nudos_aprox", 0),
                    riesgo_contacto_persona=poste.get("riesgo_contacto_persona", False),
                )
            except Exception as exc:
                self._mostrar_error(
                    f"El analisis se completo, pero no se pudo guardar en el historial: {exc}",
                    silencioso=True)

            self._ultimo_registro = {
                "tipo": poste["tipo"],
                "cables_min": poste["cables_min"],
                "cables_max": poste["cables_max"],
                "estado": poste["estado"],
                "recomendacion": poste["recomendacion"],
                "descripcion": poste.get("mensaje"),
                "zona": self.zona_actual,
                "notas_usuario": notas_usuario,
                "notas_pulidas": notas_pulidas,
                "nudo_detectado": poste["nudo_detectado"],
                "num_nudos_aprox": poste.get("num_nudos_aprox", 0),
                "riesgo_contacto_persona": poste.get("riesgo_contacto_persona", False),
                "fecha": time.strftime("%d/%m/%Y %H:%M"),
            }

            if "PELIGRO" in poste["estado"] or "ALERTA" in poste["estado"] or poste.get("riesgo_contacto_persona"):
                self._ofrecer_avisar(motivo_persona=poste.get("riesgo_contacto_persona", False))

            # NUEVO: si la foto cargada es una de las 18 del banco
            # verificado a mano, se muestra la comparacion esperado
            # vs. obtenido -- util para la demo, y deja explicito que
            # una respuesta es la referencia humana y la otra es la
            # salida real del sistema (no es la misma cosa).
            caso_verificado = buscar_caso_verificado(self._nombre_archivo_actual)
            if caso_verificado:
                messagebox.showinfo("Comparacion con banco verificado", texto_comparacion(caso_verificado, poste))
        else:
            self._mostrar_error(f"Ocurrio un problema durante el analisis:\n{payload}")

    def _pedir_nota_campo(self):
        nota = simpledialog.askstring(
            "Nota de campo (opcional)",
            "¿Algo que agregar sobre lo que ves? Ej: 'esta lloviendo, un "
            "cable esta chispeando cerca de la base'.\nDejalo vacio si no "
            "hay nada que anotar.",
            parent=self.root,
        )
        return (nota or "").strip()

    def _guardar_temp_imagen_completa(self, imagen, nombre_archivo):
        os.makedirs("temp", exist_ok=True)
        ruta = os.path.join("temp", nombre_archivo)
        cv2.imwrite(ruta, imagen)
        return ruta

    def _set_botones_ocupados(self, ocupado):
        estado = "disabled" if ocupado else "normal"
        self.btn_cargar.configure(state=estado)
        self.btn_camara.configure(state=estado)

    # ---------- ALERTA A TERCEROS ----------
    def _ofrecer_avisar(self, motivo_persona=False):
        # NUEVO: si el resultado sale en PELIGRO o ALERTA, se ofrece
        # avisar por WhatsApp a alguien (familiar, vecino, grupo del
        # barrio) para que evite la zona. No manda nada automatico —
        # abre WhatsApp con el mensaje ya escrito y el usuario decide
        # si lo envia.
        if not messagebox.askyesno(
                "Alerta de riesgo",
                "Se detecto un poste en estado de riesgo.\n"
                "¿Deseas avisar a alguien por WhatsApp para que evite la zona?"):
            return
        numero = simpledialog.askstring(
            "Numero de WhatsApp",
            "Numero con codigo de pais, sin '+' (ej: 51987654321).\n"
            "Dejalo vacio para cancelar.",
            parent=self.root,
        )
        if not numero:
            return
        zona_txt = self.zona_actual or "una zona cercana"
        if motivo_persona:
            mensaje = (
                f"Aviso S.A.I.R.E.: se detecto una persona a corta distancia de "
                f"cables o un poste en riesgo en {zona_txt}. Avisale que se aleje."
            )
        else:
            mensaje = (
                f"Aviso S.A.I.R.E.: se detecto un poste en estado de riesgo "
                f"en {zona_txt}. Evita acercarte si puedes."
            )
        url = f"https://wa.me/{numero.strip()}?text={urllib.parse.quote(mensaje)}"
        webbrowser.open(url)

    # ---------- VISUALIZACION ----------
    def _cv2_a_imagetk(self, img_bgr, size):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize(size)
        return ImageTk.PhotoImage(img_pil)

    def _mostrar_imagenes(self, original, procesada):
        self.lbl_original.imgtk = self._cv2_a_imagetk(original, (480, 360))
        self.lbl_original.configure(image=self.lbl_original.imgtk, text="")
        self.lbl_procesada.imgtk = self._cv2_a_imagetk(procesada, (480, 360))
        self.lbl_procesada.configure(image=self.lbl_procesada.imgtk, text="")

    def _actualizar_estado(self, poste):
        color_bgr = poste["color_bgr"]
        color_hex = "#%02x%02x%02x" % (color_bgr[2], color_bgr[1], color_bgr[0])
        texto = (
            f"{poste['estado']}   |   {poste['tipo']}   |   "
            f"Cables: {poste['cables_min']}-{poste['cables_max']} (estimado)"
        )
        extras = []
        if poste.get("nudo_detectado"):
            extras.append(f"Nudos aprox.: {poste.get('num_nudos_aprox', 0)}")
        if poste.get("riesgo_contacto_persona"):
            extras.append("⚠ Persona cerca")
        if poste.get("cable_bajo_detectado"):
            extras.append("⚠ Cable colgando bajo")
        if extras:
            texto += "   |   " + "   |   ".join(extras)
        self.lbl_estado.configure(text=texto, fg=color_hex)

    def _mostrar_error(self, mensaje, silencioso=False):
        self.lbl_estado.configure(text=mensaje.splitlines()[0], fg=ROJO)
        if not silencioso:
            messagebox.showerror("S.A.I.R.E. PRO", mensaje)

    # ---------- EXPORTAR IMAGEN ----------
    def exportar_resultado(self):
        if self._ultima_imagen_procesada is None:
            messagebox.showinfo("S.A.I.R.E. PRO", "Todavia no hay ningun resultado para exportar.")
            return
        ruta = filedialog.asksaveasfilename(
            title="Exportar Resultado", defaultextension=".jpg",
            filetypes=[("Imagen JPG", "*.jpg"), ("Imagen PNG", "*.png")])
        if not ruta:
            return
        try:
            cv2.imwrite(ruta, self._ultima_imagen_procesada)
        except Exception as exc:
            self._mostrar_error(f"No se pudo exportar el resultado:\n{exc}")
            return
        messagebox.showinfo("S.A.I.R.E. PRO", f"Resultado exportado a:\n{ruta}")

    # ---------- PDF: INFORME INDIVIDUAL ----------
    def _generar_pdf_individual(self):
        if not self._ultimo_registro:
            messagebox.showinfo("S.A.I.R.E. PRO", "Todavia no hay ningun resultado para generar el informe.")
            return

        nombre_caso = simpledialog.askstring(
            "Nombre del caso (opcional)",
            "¿Como quieres identificar este caso? Ej: 'Poste esquina "
            "colegio Innova'. Dejalo vacio para usar un ID automatico.",
            parent=self.root,
        )

        try:
            import reportes  # import diferido: si falta fpdf2, el resto de la app sigue funcionando
            ruta = reportes.generar_pdf_individual(
                self._ultimo_registro,
                ruta_imagen_procesada=self._ultima_ruta_imagen_completa,
                ruta_imagen_original=self._ultima_ruta_imagen_original,
                nombre_caso=nombre_caso,
            )
        except ImportError:
            messagebox.showerror("S.A.I.R.E. PRO", "Falta instalar fpdf2.\nEjecuta: pip install fpdf2")
            return
        except Exception as exc:
            self._mostrar_error(f"No se pudo generar el informe:\n{exc}")
            return

        reportes.abrir_pdf(ruta)  # lo abre solo, para que se sienta como "descarga automatica"
        messagebox.showinfo("S.A.I.R.E. PRO", f"Informe generado y abierto:\n{ruta}")
        self._ofrecer_enviar_correo(ruta)

    def _generar_pdf_desde_registro(self, registro):
        try:
            import reportes
            ruta = reportes.generar_pdf_individual(registro, ruta_imagen_procesada=registro.get("miniatura"))
        except ImportError:
            messagebox.showerror("S.A.I.R.E. PRO", "Falta instalar fpdf2.\nEjecuta: pip install fpdf2")
            return
        except Exception as exc:
            self._mostrar_error(f"No se pudo generar el informe:\n{exc}")
            return
        reportes.abrir_pdf(ruta)
        messagebox.showinfo("S.A.I.R.E. PRO", f"Informe generado y abierto:\n{ruta}")

    def _ofrecer_enviar_correo(self, ruta_pdf):
        # NOTA IMPORTANTE: ni mailto: ni el enlace web de WhatsApp
        # permiten adjuntar archivos automaticamente (limitacion de
        # ambos protocolos, no de esta app). Lo que SI se puede
        # automatizar es dejar el mensaje ya redactado y la carpeta del
        # PDF abierta, para que adjuntarlo sea un arrastrar-y-soltar
        # en vez de escribir todo desde cero. S.A.I.R.E. se encarga de
        # redactar el documento; el envio final queda en manos del
        # usuario, que es quien elige el canal y a quien se lo manda.
        opcion = messagebox.askyesnocancel(
            "Enviar informe",
            "¿Como quieres enviar este informe?\n\n"
            "Si -> Correo electronico\nNo -> WhatsApp\nCancelar -> No enviar ahora",
        )
        if opcion is None:
            return

        if opcion:
            asunto = "Reporte de riesgo electrico - S.A.I.R.E. PRO"
            cuerpo = (
                f"Se detecto un poste en estado de riesgo en la zona: "
                f"{self.zona_actual or '(no especificada)'}.\n\n"
                f"Informe adjunto (recuerda adjuntar el archivo: {ruta_pdf}).\n\n"
                "Generado con S.A.I.R.E. PRO."
            )
            url = f"mailto:?subject={urllib.parse.quote(asunto)}&body={urllib.parse.quote(cuerpo)}"
            webbrowser.open(url)
        else:
            numero = simpledialog.askstring(
                "Numero de WhatsApp",
                "Numero con codigo de pais, sin '+' (ej: 51987654321).\n"
                "Dejalo vacio para abrir WhatsApp sin numero fijo.",
                parent=self.root,
            )
            mensaje = (
                f"Reporte S.A.I.R.E. PRO: poste en estado de riesgo en "
                f"{self.zona_actual or 'zona no especificada'}. "
                f"Informe adjunto (recuerda adjuntar el archivo: {ruta_pdf})."
            )
            base = f"https://wa.me/{numero.strip()}" if numero else "https://wa.me/"
            url = f"{base}?text={urllib.parse.quote(mensaje)}"
            webbrowser.open(url)

        try:
            os.startfile(os.path.dirname(os.path.abspath(ruta_pdf)))  # abre la carpeta en Windows
        except Exception:
            pass  # en Mac/Linux el usuario navega manualmente a la carpeta reportes/

    # ---------- RESUMEN / PDF AGREGADO ----------
    def _mostrar_resumen(self):
        resumen = resumen_historial()
        if resumen["total"] == 0:
            messagebox.showinfo("S.A.I.R.E. PRO", "Todavia no hay registros en el historial.")
            return
        lineas = [f"Total de postes auditados: {resumen['total']}", ""]
        lineas.append("Por estado:")
        for estado, cantidad in resumen["por_estado"].items():
            lineas.append(f"  - {estado}: {cantidad}")
        lineas.append("")
        lineas.append("Por zona:")
        for zona, cantidad in resumen["por_zona"].items():
            lineas.append(f"  - {zona}: {cantidad}")
        lineas.append("")
        lineas.append(f"Casos con posible conexion no regulada: {resumen['con_nudo']}")
        messagebox.showinfo("Resumen de Auditorias", "\n".join(lineas))

    def _generar_pdf_resumen(self):
        registros = cargar_historial()
        resumen = resumen_historial()
        if resumen["total"] == 0:
            messagebox.showinfo("S.A.I.R.E. PRO", "Todavia no hay registros en el historial.")
            return
        try:
            import reportes
            ruta = reportes.generar_pdf_resumen(registros, resumen)
        except ImportError:
            messagebox.showerror("S.A.I.R.E. PRO", "Falta instalar fpdf2.\nEjecuta: pip install fpdf2")
            return
        except Exception as exc:
            self._mostrar_error(f"No se pudo generar el resumen:\n{exc}")
            return
        messagebox.showinfo("S.A.I.R.E. PRO", f"Resumen generado:\n{ruta}")

    # ---------- HISTORIAL ----------
    def abrir_historial(self):
        ventana = tk.Toplevel(self.root)
        ventana.title("Historial de Auditorias")
        ventana.configure(bg=BG)
        ventana.geometry("500x600")
        ventana.minsize(420, 320)

        barra_superior = tk.Frame(ventana, bg=BG)
        barra_superior.pack(fill="x", padx=10, pady=(10, 0))
        btn_vaciar = tk.Button(
            barra_superior, text="Vaciar historial", bg=ROJO, fg="#101820",
            font=("Segoe UI", 9, "bold"), relief="flat", padx=10, pady=4, cursor="hand2",
            command=lambda: self._confirmar_vaciar_historial(ventana))
        btn_vaciar.pack(side="right")

        contenedor_scroll = tk.Frame(ventana, bg=BG)
        contenedor_scroll.pack(fill="both", expand=True, padx=10, pady=10)

        canvas = tk.Canvas(contenedor_scroll, bg=BG, highlightthickness=0)
        scroll = tk.Scrollbar(contenedor_scroll, orient="vertical", command=canvas.yview)
        contenedor = tk.Frame(canvas, bg=BG)
        contenedor.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=contenedor, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self._dibujar_lista_historial(contenedor, ventana)

    def _dibujar_lista_historial(self, contenedor, ventana_historial):
        for hijo in contenedor.winfo_children():
            hijo.destroy()

        registros = cargar_historial()
        if not registros:
            tk.Label(contenedor, text="Aun no hay auditorias guardadas.", bg=BG, fg=TEXTO).pack(pady=20)
            return

        self._miniaturas_historial = []
        for indice, r in enumerate(registros):
            fila = tk.Frame(contenedor, bg=PANEL)
            fila.pack(fill="x", padx=8, pady=4)

            ruta_miniatura = r.get("miniatura")
            if ruta_miniatura and os.path.exists(ruta_miniatura):
                try:
                    img = Image.open(ruta_miniatura)
                    imgtk = ImageTk.PhotoImage(img)
                    self._miniaturas_historial.append(imgtk)
                    tk.Label(fila, image=imgtk, bg=PANEL).pack(side="left", padx=6, pady=6)
                except Exception:
                    tk.Label(fila, text="[imagen no disponible]", bg=PANEL, fg=TEXTO_SEC,
                              width=16).pack(side="left", padx=6, pady=6)
            else:
                tk.Label(fila, text="[imagen no disponible]", bg=PANEL, fg=TEXTO_SEC,
                          width=16).pack(side="left", padx=6, pady=6)

            cables_txt = f'{r.get("cables_min", "?")}-{r.get("cables_max", "?")}'
            zona_txt = r.get("zona") or "sin zona"
            texto = (
                f'{r.get("fecha", "?")}  ({zona_txt})\n'
                f'{r.get("tipo", "?")} - {cables_txt} cables (est.)\n'
                f'{r.get("estado", "?")}'
            )
            tk.Label(fila, text=texto, bg=PANEL, fg=TEXTO, justify="left").pack(side="left", padx=8)

            btn_pdf_fila = tk.Button(
                fila, text="PDF", bg=PANEL_ALT, fg=ACENTO, relief="flat",
                font=("Segoe UI", 8, "bold"), cursor="hand2", padx=8, pady=4,
                command=lambda reg=r: self._generar_pdf_desde_registro(reg))
            btn_pdf_fila.pack(side="right", padx=6)

            btn_borrar = tk.Button(
                fila, text="Eliminar", bg=PANEL_ALT, fg=ROJO, relief="flat",
                font=("Segoe UI", 8, "bold"), cursor="hand2", padx=8, pady=4,
                command=lambda i=indice: self._eliminar_entrada_historial(i, contenedor, ventana_historial))
            btn_borrar.pack(side="right", padx=6)

    def _eliminar_entrada_historial(self, indice, contenedor, ventana_historial):
        eliminar_registro(indice)
        self._dibujar_lista_historial(contenedor, ventana_historial)

    def _confirmar_vaciar_historial(self, ventana_historial):
        if messagebox.askyesno(
                "Vaciar historial",
                "Esto borrara todos los registros y sus imagenes guardadas. "
                "Esta accion no se puede deshacer.\n\n¿Continuar?"):
            vaciar_historial()
            ventana_historial.destroy()
            self.abrir_historial()


if __name__ == "__main__":
    root = tk.Tk()
    app = AppSaire(root)
    root.mainloop()
