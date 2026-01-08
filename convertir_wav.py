from pydub import AudioSegment
import os

def convertir_a_wav(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    formatos_permitidos = (".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma")

    for archivo in os.listdir(input_folder):
        nombre, ext = os.path.splitext(archivo)
        if ext.lower() in formatos_permitidos:
            ruta_entrada = os.path.join(input_folder, archivo)
            ruta_salida = os.path.join(output_folder, f"{nombre}.wav")

            try:
                audio = AudioSegment.from_file(ruta_entrada)
                audio.export(ruta_salida, format="wav")
                print(f"✅ Convertido: {archivo} → {nombre}.wav")
            except Exception as e:
                print(f"❌ Error al convertir {archivo}: {e}")

# 🧪 Ejemplo de uso
convertir_a_wav("audios_originales", "audios_convertidos")
