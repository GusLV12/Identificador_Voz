
import sounddevice as sd
import soundfile as sf
import time
import numpy as np

def grabar_con_cuenta_regresiva(duracion, archivo_salida):
    print("Preparando grabación...")
    for i in range(3, 0, -1):
        print(f"Grabando en {i}...")
        time.sleep(1)

    print("¡Grabando!")
    fs = 44100  # Frecuencia de muestreo
    grabacion = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    sf.write(archivo_salida, grabacion, fs)
    print(f"Grabación guardada en {archivo_salida}")
