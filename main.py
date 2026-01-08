# main.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from prediccion import predecir_vocal
from ui_theme import apply_theme, COLORS

# Compatibilidad librosa + numpy
np.complex = complex

# Configuración
DURACION = 5
FS = 44100
AUDIO_PATH = "audios/temp_audio.wav"


class AnalizadorVozApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Analizador de Señales de Voz")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 750)

        apply_theme(self.root)

        self.construir_interfaz()
        self.actualizar_estado("Sistema listo. Esperando adquisición de audio.", COLORS["text"])

    # ───────────────────────── INTERFAZ ─────────────────────────

    def construir_interfaz(self):
        self.contenedor_principal = tk.Frame(self.root, bg=COLORS["bg"])
        self.contenedor_principal.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.contenedor_principal.columnconfigure(0, weight=0)
        self.contenedor_principal.columnconfigure(1, weight=1)

        self.crear_panel_control()
        self.crear_dashboard()

    def crear_panel_control(self):
        panel = tk.Frame(
            self.contenedor_principal,
            bg=COLORS["panel"],
            padx=12,
            pady=12
        )
        panel.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        ttk.Label(
            panel,
            text="ANALIZADOR DE VOZ",
            style="Title.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            panel,
            text="Extracción y clasificación de características acústicas",
            style="Subtitle.TLabel"
        ).pack(anchor="w", pady=(0, 15))

        self.lbl_estado = tk.Label(
            panel,
            text="",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Consolas", 10),
            justify="left",
            wraplength=260
        )
        self.lbl_estado.pack(fill=tk.X, pady=10)

        ttk.Button(
            panel,
            text="Grabar audio (5 segundos)",
            style="Panel.TButton",
            command=self.iniciar_grabacion
        ).pack(fill=tk.X, pady=5)

        ttk.Button(
            panel,
            text="Reproducir audio",
            style="Panel.TButton",
            command=self.reproducir_audio
        ).pack(fill=tk.X, pady=5)

        ttk.Button(
            panel,
            text="Información del sistema",
            style="Panel.TButton",
            command=self.mostrar_info
        ).pack(fill=tk.X, pady=5)

    def crear_dashboard(self):
        dash = tk.Frame(self.contenedor_principal, bg=COLORS["bg"])
        dash.grid(row=0, column=1, sticky="nsew")

        dash.columnconfigure(0, weight=1)
        dash.columnconfigure(1, weight=1)
        dash.rowconfigure(0, weight=1)
        dash.rowconfigure(1, weight=1)

        self.card_wave = self.crear_tarjeta(dash, "Señal en el dominio del tiempo", 0, 0)
        self.card_fft = self.crear_tarjeta(dash, "Espectro de frecuencias", 0, 1)
        self.card_spec = self.crear_tarjeta(dash, "Espectrograma (STFT)", 1, 0, colspan=2)
        self.card_out = self.crear_tarjeta(dash, "Resultado de la clasificación", 2, 0, colspan=2, height=160)

    def crear_tarjeta(self, parent, titulo, fila, columna, colspan=1, height=None):
        frame = tk.LabelFrame(
            parent,
            text=titulo,
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10, "bold"),
            bd=1,
            relief="solid",
            padx=10,
            pady=10
        )
        frame.grid(row=fila, column=columna, columnspan=colspan, sticky="nsew", padx=8, pady=8)
        if height:
            frame.config(height=height)
            frame.grid_propagate(False)
        return frame

    # ───────────────────────── ESTADO ─────────────────────────

    def actualizar_estado(self, texto, color):
        self.lbl_estado.config(text=texto, fg=color)

    # ───────────────────────── AUDIO ─────────────────────────

    def iniciar_grabacion(self):
        threading.Thread(target=self.grabar_y_analizar, daemon=True).start()

    def grabar_y_analizar(self):
        self.actualizar_estado("Grabando audio...\nHable claramente frente al micrófono.", COLORS["muted"])

        try:
            audio = sd.rec(
                int(DURACION * FS),
                samplerate=FS,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            os.makedirs("audios", exist_ok=True)
            sf.write(AUDIO_PATH, audio, FS)

            self.root.after(0, self.actualizar_graficas)
            self.root.after(0, self.ejecutar_prediccion)

            self.actualizar_estado("Análisis completado correctamente.", COLORS["ok"])

        except Exception as e:
            self.actualizar_estado(f"Error durante la grabación:\n{e}", COLORS["err"])
            messagebox.showerror("Error de grabación", str(e))

    def reproducir_audio(self):
        if not os.path.exists(AUDIO_PATH):
            messagebox.showwarning("Sin audio", "No hay audio grabado.")
            return

        y, sr = sf.read(AUDIO_PATH, dtype="float32")
        sd.play(y, sr)
        sd.wait()

    # ───────────────────────── GRÁFICAS ─────────────────────────

    def actualizar_graficas(self):
        y, sr = librosa.load(AUDIO_PATH, sr=None)

        self.graficar_onda(y, sr)
        self.graficar_fft(y, sr)
        self.graficar_espectrograma(y, sr)

    def graficar_onda(self, y, sr):
        self.limpiar(self.card_wave)
        fig, ax = plt.subplots(figsize=(6, 3), dpi=90)
        ax.plot(y, linewidth=0.6)
        ax.set_title("Amplitud de la señal")
        canvas = FigureCanvasTkAgg(fig, master=self.card_wave)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def graficar_fft(self, y, sr):
        self.limpiar(self.card_fft)
        Y = np.abs(np.fft.rfft(y))
        f = np.fft.rfftfreq(len(y), 1 / sr)

        fig, ax = plt.subplots(figsize=(6, 3), dpi=90)
        ax.plot(f[f < 4000], Y[f < 4000], linewidth=0.6)
        ax.set_title("Espectro de magnitud (0–4 kHz)")
        canvas = FigureCanvasTkAgg(fig, master=self.card_fft)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def graficar_espectrograma(self, y, sr):
        self.limpiar(self.card_spec)
        S = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=np.max)

        fig, ax = plt.subplots(figsize=(12, 3), dpi=90)
        librosa.display.specshow(S, sr=sr, cmap="gray_r", ax=ax)
        ax.set_title("Espectrograma en dB")

        canvas = FigureCanvasTkAgg(fig, master=self.card_spec)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ───────────────────────── RESULTADOS ─────────────────────────

    def ejecutar_prediccion(self):
        self.limpiar(self.card_out)

        try:
            r = predecir_vocal(AUDIO_PATH)

            texto = (
                f"Etiqueta     : {r.get('etiqueta')}\n"
                f"Tipo         : {r.get('tipo')}\n"
                f"Género       : {r.get('genero')}\n"
                f"Tipo de voz  : {r.get('voz')}\n"
                f"Confianza    : {r.get('confianza', 0)}%"
            )

            tk.Label(
                self.card_out,
                text=texto,
                font=("Consolas", 10),
                bg=COLORS["card"],
                fg=COLORS["text"],
                justify="left"
            ).pack(anchor="w")

        except Exception as e:
            tk.Label(
                self.card_out,
                text=str(e),
                fg=COLORS["err"],
                bg=COLORS["card"]
            ).pack(anchor="w")

    # ───────────────────────── UTILIDADES ─────────────────────────

    def limpiar(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    def mostrar_info(self):
        messagebox.showinfo(
            "Información del sistema",
            "Analizador de señales de voz\n\n"
            "Características extraídas:\n"
            "- MFCC\n- Estadísticas espectrales\n- Formantes\n\n"
            "Asegúrese de entrenar los modelos antes de ejecutar."
        )


# ───────────────────────── MAIN ─────────────────────────

if __name__ == "__main__":

    modelos_requeridos = [
        "modelos/scaler.pkl",
        "modelos/encoder.pkl",
        "modelos/modelo_formantes.pkl",
        "modelos/modelo_genero.pkl",
        "modelos/modelo_voz.pkl"
    ]

    faltantes = [m for m in modelos_requeridos if not os.path.exists(m)]

    if faltantes:
        messagebox.showerror(
            "Modelos faltantes",
            "No se encontraron los siguientes modelos:\n\n" + "\n".join(faltantes)
        )
    else:
        root = tk.Tk()
        AnalizadorVozApp(root)
        root.mainloop()
