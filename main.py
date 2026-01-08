import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display
import os
from prediccion import predecir_vocal

np.complex = complex  # Compatibilidad librosa + numpy

DURACION = 5
AUDIO_PATH = "audios/temp_audio.wav"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Identificador de Vocales y Sílabas - Versión Mejorada")
        self.root.geometry("1450x950")  # Aumentado ligeramente para mejor visualización
        self.root.configure(bg='white')  # Fondo blanco como solicitas
        self.root.resizable(True, True)  # Permitir redimensionar
        
        # Configurar estilo
        self.setup_style()
        
        # Frame principal con fondo blanco
        self.main_frame = tk.Frame(root, bg='white')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header con título y controles
        self.create_header()
        
        # Frame para las gráficas (2x2)
        self.create_graphics_frame()
        
        # Frame para información adicional
        self.create_info_frame()

    def setup_style(self):
        """Configurar estilos personalizados"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Estilo para botones principales - NEGRO CON LETRAS BLANCAS
        style.configure('Custom.TButton', 
                       font=('Arial', 11, 'bold'),
                       padding=(20, 10),
                       background='#2c2c2c',
                       foreground='white',
                       borderwidth=1,
                       relief='solid')
        
        style.map('Custom.TButton',
                 background=[('active', '#404040'),
                           ('pressed', '#1a1a1a')])
        
        # Estilo para botones secundarios - NEGRO CON LETRAS BLANCAS
        style.configure('Success.TButton',
                       font=('Arial', 10, 'bold'),
                       padding=(15, 8),
                       background='#2c2c2c',
                       foreground='white',
                       borderwidth=1,
                       relief='solid')
        
        style.map('Success.TButton',
                 background=[('active', '#404040'),
                           ('pressed', '#1a1a1a')])
        
        # Estilo para labels con fondo blanco
        style.configure('Title.TLabel',
                       font=('Arial', 16, 'bold'),
                       background='white',
                       foreground='#2E86AB')
        
        style.configure('Subtitle.TLabel',
                       font=('Arial', 12),
                       background='white')
        
        style.configure('Status.TLabel',
                       font=('Arial', 11),
                       background='white')

    def create_header(self):
        """Crear el header con título y controles"""
        header_frame = tk.Frame(self.main_frame, bg='white')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Título principal
        title_label = ttk.Label(header_frame, 
                               text="🎤 Identificador de Vocales y Sílabas L", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 5))
        
        # Subtítulo con información del sistema
        subtitle_label = ttk.Label(header_frame,
                                  text="Procesamiento Digital de Señales",
                                  style='Subtitle.TLabel')
        subtitle_label.pack(pady=(0, 15))
        
        # Frame para controles
        controls_frame = tk.Frame(header_frame, bg='white')
        controls_frame.pack()
        
        # Estado
        self.lbl_estado = ttk.Label(controls_frame, 
                                   text="⏺ Esperando grabación...", 
                                   style='Status.TLabel')
        self.lbl_estado.pack(pady=5)
        
        # Botones con estilo negro
        buttons_frame = tk.Frame(controls_frame, bg='white')
        buttons_frame.pack(pady=10)
        
        self.btn_grabar = ttk.Button(buttons_frame, 
                                    text="🎧 Grabar Audio (5s)", 
                                    command=self.iniciar_grabacion,
                                    style='Custom.TButton')
        self.btn_grabar.pack(side=tk.LEFT, padx=5)
        
        self.btn_reproducir = ttk.Button(buttons_frame, 
                                        text="▶️ Reproducir", 
                                        command=self.reproducir_audio,
                                        style='Success.TButton')
        self.btn_reproducir.pack(side=tk.LEFT, padx=5)
        
        # Botón de información
        self.btn_info = ttk.Button(buttons_frame,
                                  text="ℹ️ Info",
                                  command=self.mostrar_info_sistema,
                                  style='Success.TButton')
        self.btn_info.pack(side=tk.LEFT, padx=5)

    def create_graphics_frame(self):
        """Crear el frame para las gráficas en formato 2x2"""
        # Frame contenedor para las gráficas con fondo blanco
        self.graphics_container = tk.Frame(self.main_frame, bg='white')
        self.graphics_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Configurar grid 2x2
        self.graphics_container.grid_columnconfigure(0, weight=1)
        self.graphics_container.grid_columnconfigure(1, weight=1)
        self.graphics_container.grid_rowconfigure(0, weight=1)
        self.graphics_container.grid_rowconfigure(1, weight=1)
        
        # Frames para cada gráfica con mejor espaciado
        self.frame_waveform = self.create_graph_frame(0, 0, "📈 Forma de Onda")
        self.frame_fft = self.create_graph_frame(0, 1, "🔬 Transformada de Fourier")
        self.frame_spectrogram = self.create_graph_frame(1, 0, "🌈 Espectrograma")
        self.frame_results = self.create_graph_frame(1, 1, "🎯 Resultados del Análisis")

    def create_graph_frame(self, row, col, title):
        """Crear un frame individual para cada gráfica"""
        frame = tk.LabelFrame(self.graphics_container, 
                             text=title, 
                             bg='white',
                             font=('Arial', 11, 'bold'),
                             fg='#333',
                             bd=2,
                             relief='groove',
                             padx=8, 
                             pady=8)
        frame.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')
        return frame

    def create_info_frame(self):
        """Crear frame para información adicional"""
        self.info_frame = tk.Frame(self.main_frame, bg='white')
        self.info_frame.pack(fill=tk.X)
        
        info_label = ttk.Label(self.info_frame, 
                              text="💡 Habla claramente durante 5 segundos • El sistema analiza 20 características acústicas",
                              style='Subtitle.TLabel')
        info_label.pack()

    def mostrar_info_sistema(self):
        """Mostrar información del sistema"""
        try:
            import joblib
            metadata = joblib.load("modelos/metadata.pkl")
            resultados = metadata.get('resultados_entrenamiento', {})
            
            info_text = "🤖 INFORMACIÓN DEL SISTEMA\n\n"
            info_text += f"📊 Muestras de entrenamiento: {metadata.get('n_samples', 'N/A')}\n"
            info_text += f"🔧 Características: {metadata.get('n_features', 'N/A')}\n"
            info_text += f"🏷️ Etiquetas: {len(metadata.get('etiquetas_unicas', []))}\n\n"
            
            info_text += "📈 PRECISIÓN DE MODELOS:\n"
            for modelo, stats in resultados.items():
                precision = stats.get('cv_mean', 0) * 100
                info_text += f"• {modelo.capitalize()}: {precision:.1f}%\n"
            
            info_text += f"\n🎵 Etiquetas disponibles:\n{', '.join(metadata.get('etiquetas_unicas', []))}"
            
        except Exception:
            info_text = "🤖 SISTEMA DE ANÁLISIS DE VOZ\n\n"
            info_text += "✨ Características analizadas:\n"
            info_text += "• F0 (frecuencia fundamental)\n"
            info_text += "• F1, F2, F3, F4 (formantes)\n"
            info_text += "• Centroide espectral\n"
            info_text += "• MFCC (coeficientes cepstrales)\n"
            info_text += "• Características temporales\n"
            info_text += "• Análisis prosódico"
        
        messagebox.showinfo("Información del Sistema", info_text)

    def iniciar_grabacion(self):
        self.btn_grabar.config(state=tk.DISABLED)
        self.btn_reproducir.config(state=tk.DISABLED)
        threading.Thread(target=self.grabar_y_procesar).start()

    def grabar_y_procesar(self):
        # Countdown
        for i in range(3, 0, -1):
            self.lbl_estado.config(text=f"⏳ Grabación comienza en {i}...")
            self.root.update()
            threading.Event().wait(1)

        self.lbl_estado.config(text="🎧 Grabando... Habla claramente")
        fs = 44100
        try:
            grabacion = sd.rec(int(DURACION * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(AUDIO_PATH), exist_ok=True)
            sf.write(AUDIO_PATH, grabacion, fs)
            
        except Exception as e:
            self.lbl_estado.config(text="❌ Error al grabar")
            messagebox.showerror("Error de Grabación", str(e))
            self.btn_grabar.config(state=tk.NORMAL)
            return

        self.lbl_estado.config(text="✅ Grabación finalizada. Procesando...")
        self.mostrar_graficas()
        self.mostrar_resultado()
        self.lbl_estado.config(text="🎯 Análisis completado con éxito")
        self.btn_grabar.config(state=tk.NORMAL)
        self.btn_reproducir.config(state=tk.NORMAL)

    def limpiar_frame(self, frame):
        """Limpiar contenido de un frame"""
        for widget in frame.winfo_children():
            widget.destroy()

    def mostrar_graficas(self):
        """Mostrar las gráficas en el layout 2x2 con explicaciones mejoradas"""
        try:
            y, sr = librosa.load(AUDIO_PATH, sr=None)
            
            # 1. Forma de onda (superior izquierda)
            self.limpiar_frame(self.frame_waveform)
            fig1, ax1 = plt.subplots(figsize=(6, 3.0), facecolor='white', dpi=80)
            librosa.display.waveshow(y, sr=sr, ax=ax1, color='#2E86AB', alpha=0.8)
            ax1.set_title("Señal de Audio", fontsize=11, fontweight='bold', color='#333')
            ax1.set_xlabel("Tiempo (s)", fontsize=9)
            ax1.set_ylabel("Amplitud", fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=8)
            plt.tight_layout(pad=1.0)
            
            canvas1 = FigureCanvasTkAgg(fig1, master=self.frame_waveform)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            desc1 = tk.Label(self.frame_waveform, 
                           text="📊 Muestra cómo varía la intensidad de tu voz en el tiempo.\nLos picos indican momentos de mayor energía vocal.",
                           font=("Arial", 8), bg='white', fg='#666',
                           justify=tk.CENTER)
            desc1.pack(pady=(2, 5))

            # 2. Transformada de Fourier (superior derecha)
            self.limpiar_frame(self.frame_fft)
            Y = np.abs(np.fft.rfft(y))
            freqs = np.fft.rfftfreq(len(y), 1/sr)
            
            fig2, ax2 = plt.subplots(figsize=(6, 3.0), facecolor='white', dpi=80)
            ax2.plot(freqs[:len(freqs)//4], Y[:len(Y)//4], color='#A23B72', linewidth=1.5)
            ax2.set_title("Análisis de Frecuencias", fontsize=11, fontweight='bold', color='#333')
            ax2.set_xlabel("Frecuencia (Hz)", fontsize=9)
            ax2.set_ylabel("Magnitud", fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=8)
            
            # Marcar formantes típicos
            ax2.axvline(x=500, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax2.axvline(x=1500, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            ax2.text(500, max(Y[:len(Y)//4])*0.8, 'F1', fontsize=8, color='red')
            ax2.text(1500, max(Y[:len(Y)//4])*0.8, 'F2', fontsize=8, color='orange')
            
            plt.tight_layout(pad=1.0)
            
            canvas2 = FigureCanvasTkAgg(fig2, master=self.frame_fft)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            desc2 = tk.Label(self.frame_fft, 
                           text="🔬 Análisis de frecuencias. Los picos indican\nlas frecuencias dominantes (F0, formantes).",
                           font=("Arial", 8), bg='white', fg='#666',
                           justify=tk.CENTER)
            desc2.pack(pady=(2, 5))

            # 3. Espectrograma (inferior izquierda)
            self.limpiar_frame(self.frame_spectrogram)
            fig3, ax3 = plt.subplots(figsize=(6, 3.0), facecolor='white', dpi=80)
            S = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', 
                                         cmap='viridis', ax=ax3)
            ax3.set_title("Mapa Tiempo-Frecuencia", fontsize=11, fontweight='bold', color='#333')
            ax3.set_xlabel("Tiempo (s)", fontsize=9)
            ax3.set_ylabel("Frecuencia (Hz)", fontsize=9)
            ax3.tick_params(labelsize=8)
            
            # Colorbar más pequeño
            cbar = plt.colorbar(img, ax=ax3, format="%+2.0f dB", shrink=0.8)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label('dB', fontsize=8)
            plt.tight_layout(pad=1.0)
            
            canvas3 = FigureCanvasTkAgg(fig3, master=self.frame_spectrogram)
            canvas3.draw()
            canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            desc3 = tk.Label(self.frame_spectrogram, 
                           text="🌈 Mapa tiempo-frecuencia. Colores cálidos = mayor energía.\nPermite ver cómo cambian los formantes.",
                           font=("Arial", 8), bg='white', fg='#666',
                           justify=tk.CENTER)
            desc3.pack(pady=(2, 5))

        except Exception as e:
            messagebox.showerror("Error al generar gráficas", str(e))

    def mostrar_resultado(self):
        """Mostrar resultados en el cuadrante inferior derecho - DISEÑO MEJORADO"""
        self.limpiar_frame(self.frame_results)
        
        try:
            resultado = predecir_vocal(AUDIO_PATH)
            
            # Crear contenedor principal con scroll si es necesario
            result_container = tk.Frame(self.frame_results, bg='white')
            result_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            
            # Título de resultados
            title_result = tk.Label(result_container, 
                                  text="🧬 Análisis Completado", 
                                  font=("Arial", 13, "bold"), 
                                  bg='white', fg='#2E86AB')
            title_result.pack(pady=(0, 10))
            
            # Resultados principales en formato más compacto
            results_data = [
                ("📊 Tipo:", resultado.get('tipo', 'N/A')),
                ("🏷️ Detectado:", resultado.get('etiqueta', 'N/A')),
                ("👤 Género:", resultado.get('genero', 'N/A').capitalize()),
                ("🎵 Voz:", resultado.get('voz', 'N/A').capitalize())
            ]
            
            for emoji_label, value in results_data:
                result_frame = tk.Frame(result_container, bg='white')
                result_frame.pack(fill=tk.X, pady=3)
                
                # Crear frame horizontal para label y valor
                content_frame = tk.Frame(result_frame, bg='white')
                content_frame.pack(fill=tk.X)
                
                label_widget = tk.Label(content_frame, 
                                      text=emoji_label, 
                                      font=("Arial", 10, "bold"), 
                                      bg='white', fg='#333')
                label_widget.pack(side=tk.LEFT)
                
                value_widget = tk.Label(content_frame, 
                                      text=value, 
                                      font=("Arial", 10), 
                                      bg='white', fg='#2E86AB')
                value_widget.pack(side=tk.LEFT, padx=(8, 0))
            
            # CONFIANZA - Asegurar que se vea completo
            if 'confianza' in resultado:
                confianza = resultado['confianza']
                color_confianza = '#4CAF50' if confianza > 80 else '#FF9800' if confianza > 60 else '#F44336'
                
                # Separador pequeño
                sep_frame = tk.Frame(result_container, height=1, bg='#ddd')
                sep_frame.pack(fill=tk.X, pady=8)
                
                conf_frame = tk.Frame(result_container, bg='white')
                conf_frame.pack(fill=tk.X, pady=3)
                
                # Frame horizontal para confianza
                conf_content = tk.Frame(conf_frame, bg='white')
                conf_content.pack(fill=tk.X)
                
                conf_label = tk.Label(conf_content,
                                    text="🎯 Confianza:",
                                    font=("Arial", 10, "bold"),
                                    bg='white', fg='#333')
                conf_label.pack(side=tk.LEFT)
                
                conf_value = tk.Label(conf_content,
                                    text=f"{confianza}%",
                                    font=("Arial", 10, "bold"),
                                    bg='white', fg=color_confianza)
                conf_value.pack(side=tk.LEFT, padx=(8, 0))
                
                # Barra de confianza visual
                bar_frame = tk.Frame(result_container, bg='white')
                bar_frame.pack(fill=tk.X, pady=(2, 8))
                
                bar_bg = tk.Frame(bar_frame, height=6, bg='#e0e0e0')
                bar_bg.pack(fill=tk.X)
                
                bar_fill = tk.Frame(bar_bg, height=6, bg=color_confianza)
                bar_fill.place(x=0, y=0, relwidth=confianza/100)
            
            # Información técnica compacta
            info_frame = tk.Frame(result_container, bg='white')
            info_frame.pack(fill=tk.X, pady=(5, 0))
            
            
            # Espacio adicional para asegurar que todo se vea
            spacer = tk.Frame(result_container, height=10, bg='white')
            spacer.pack()
            
        except Exception as e:
            error_container = tk.Frame(self.frame_results, bg='white')
            error_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            error_title = tk.Label(error_container,
                                 text="❌ Error en Análisis",
                                 font=("Arial", 12, "bold"),
                                 bg='white', fg='#d32f2f')
            error_title.pack(pady=(0, 10))
            
            error_label = tk.Label(error_container,
                                 text=str(e),
                                 font=("Arial", 9),
                                 bg='white', fg='#d32f2f',
                                 justify=tk.CENTER,
                                 wraplength=200)
            error_label.pack()
            
            print(f"Error detallado: {e}")

    def reproducir_audio(self):
        if not os.path.exists(AUDIO_PATH):
            messagebox.showwarning("Advertencia", "No hay audio grabado aún.")
            return
        try:
            self.lbl_estado.config(text="🔊 Reproduciendo audio...")
            y, sr = sf.read(AUDIO_PATH, dtype='float32')
            if len(y) == 0:
                raise ValueError("El archivo de audio está vacío.")
            sd.play(y, sr)
            sd.wait()
            self.lbl_estado.config(text="✅ Reproducción completada")
        except Exception as e:
            messagebox.showerror("Error al reproducir", str(e))
            self.lbl_estado.config(text="❌ Error en reproducción")

if __name__ == "__main__":
    # Verificar que los modelos existan
    modelos_necesarios = [
        "modelos/scaler.pkl",
        "modelos/encoder.pkl",
        "modelos/modelo_formantes.pkl",
        "modelos/modelo_genero.pkl",
        "modelos/modelo_voz.pkl"
    ]
    
    modelos_faltantes = [m for m in modelos_necesarios if not os.path.exists(m)]
    
    if modelos_faltantes:
        print("⚠️ MODELOS FALTANTES:")
        for modelo in modelos_faltantes:
            print(f"   - {modelo}")
        print("\n🔧 Ejecuta 'python entrenar_modelos.py' primero")
        
        # Mostrar ventana de error
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Modelos No Encontrados", 
                           "Los modelos de ML no están entrenados.\n\n" +
                           "Ejecuta 'python entrenar_modelos.py' primero.")
        root.destroy()
    else:
        print("✅ Todos los modelos encontrados. Iniciando aplicación...")
        root = tk.Tk()
        app = App(root)
        root.mainloop()