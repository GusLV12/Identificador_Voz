import os
import subprocess

# Ruta base general (donde están las carpetas Silabas y Vocales)
base_path = "audios"
ext_permitida = ".wav"

# Ruta al ejecutable de ffmpeg (asegúrate que existe)
ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"

# Recorremos todas las subcarpetas
for categoria in os.listdir(base_path):
    path_categoria = os.path.join(base_path, categoria)
    if not os.path.isdir(path_categoria):
        continue

    for genero in os.listdir(path_categoria):
        path_genero = os.path.join(path_categoria, genero)
        if not os.path.isdir(path_genero):
            continue

        for etiqueta in os.listdir(path_genero):
            path_etiqueta = os.path.join(path_genero, etiqueta)
            if not os.path.isdir(path_etiqueta):
                continue

            for archivo in os.listdir(path_etiqueta):
                if archivo.lower().endswith(ext_permitida):
                    original = os.path.join(path_etiqueta, archivo)
                    temporal = os.path.join(path_etiqueta, "temp_convertido.wav")

                    comando = [
                        ffmpeg_path, "-y",
                        "-i", original,
                        "-acodec", "pcm_s16le",
                        "-ac", "1",
                        "-ar", "44100",
                        temporal
                    ]

                    try:
                        subprocess.run(comando, check=True)
                        os.remove(original)
                        os.rename(temporal, original)
                        print(f"✅ Convertido: {original}")
                    except subprocess.CalledProcessError:
                        print(f"❌ Error al convertir: {original}")
