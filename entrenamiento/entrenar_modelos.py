import os
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal
from scipy.signal import find_peaks
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURACIÓN ===
carpetas_audio = ["audios/Silabas", "audios/Vocales"]
output_csv = "dataset_entrenamiento_mejorado.csv"
output_dir = "modelos"
os.makedirs(output_dir, exist_ok=True)

print("🎯 Sistema de Entrenamiento Mejorado - Análisis de Voz")
print("=" * 60)

# === FUNCIONES DE EXTRACCIÓN MEJORADAS ===

def extraer_f0_robusto(y, sr):
    """Extracción robusta de F0"""
    try:
        # Método 1: Librosa YIN (más robusto)
        f0_series = librosa.yin(y, fmin=80, fmax=400, sr=sr)
        f0_librosa = np.nanmedian(f0_series[f0_series > 0])
        
        if not np.isnan(f0_librosa) and 80 <= f0_librosa <= 400:
            return f0_librosa
        
        # Método 2: Autocorrelación mejorada como fallback
        y_windowed = y * scipy.signal.windows.hann(len(y))
        correlation = np.correlate(y_windowed, y_windowed, mode='full')
        correlation = correlation[len(correlation)//2:]
        
        min_period = int(sr / 400)
        max_period = int(sr / 80)
        
        if max_period < len(correlation):
            peaks, properties = find_peaks(correlation[min_period:max_period], 
                                         height=0.3 * np.max(correlation[min_period:max_period]))
            if len(peaks) > 0:
                main_peak = peaks[np.argmax(properties['peak_heights'])] + min_period
                f0 = sr / main_peak
                if 80 <= f0 <= 400:
                    return f0
        
        return 150  # Valor por defecto
        
    except Exception:
        return 150

def extraer_formantes_lpc(y, sr, n_formantes=4):
    """Extracción real de formantes usando LPC"""
    try:
        # Orden LPC apropiado para voz
        order = min(int(2 + sr/1000), len(y)//4)
        
        if order < 4:  # Audio muy corto
            return [700, 1220, 2600, 3400]  # Valores por defecto
        
        # Calcular coeficientes LPC
        a = librosa.lpc(y, order=order)
        
        # Encontrar raíces y convertir a frecuencias
        roots = np.roots(a)
        angles = np.angle(roots)
        freqs = np.abs(angles) * sr / (2 * np.pi)
        
        # Filtrar frecuencias válidas
        valid_freqs = freqs[(freqs > 200) & (freqs < 4000)]
        valid_freqs = np.sort(valid_freqs)
        
        # Asegurar que tenemos 4 formantes
        formantes_defaults = [700, 1220, 2600, 3400]
        resultado = []
        
        for i in range(n_formantes):
            if i < len(valid_freqs):
                resultado.append(valid_freqs[i])
            else:
                resultado.append(formantes_defaults[i])
        
        return resultado
        
    except Exception:
        return [700, 1220, 2600, 3400]  # Fallback

def extraer_caracteristicas_completas(path):
    """Extracción completa de características acústicas"""
    try:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y[:, 0]  # Mono
        
        # Normalización
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        # Remover silencios
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y_trimmed) < sr * 0.05:  # Mínimo 50ms
            return None
        
        caracteristicas = {}
        
        # 1. F0 robusto
        caracteristicas['f0'] = extraer_f0_robusto(y_trimmed, sr)
        
        # 2. Formantes reales
        formantes = extraer_formantes_lpc(y_trimmed, sr)
        caracteristicas['f1'] = formantes[0]
        caracteristicas['f2'] = formantes[1] 
        caracteristicas['f3'] = formantes[2]
        caracteristicas['f4'] = formantes[3]
        
        # 3. Características espectrales
        centroide = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0].mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)[0].mean()
        rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)[0].mean()
        zcr = librosa.feature.zero_crossing_rate(y_trimmed)[0].mean()
        
        caracteristicas.update({
            'centroide_espectral': centroide,
            'ancho_banda': bandwidth,
            'rolloff': rolloff,
            'zcr': zcr
        })
        
        # 4. MFCC (primeros 5 coeficientes)
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=5)
        for i, mfcc_val in enumerate(np.mean(mfccs, axis=1)):
            caracteristicas[f'mfcc_{i+1}'] = mfcc_val
        
        # 5. Características temporales
        rms = librosa.feature.rms(y=y_trimmed)[0]
        caracteristicas['energia_promedio'] = np.mean(rms)
        caracteristicas['variabilidad_energia'] = np.std(rms)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y_trimmed, sr=sr, units='frames')
        caracteristicas['n_onsets'] = len(onset_frames)
        caracteristicas['duracion_efectiva'] = len(y_trimmed) / sr
        
        # 6. Características de F0
        f0_series = librosa.yin(y_trimmed, fmin=80, fmax=400, sr=sr)
        f0_valid = f0_series[f0_series > 0]
        if len(f0_valid) > 0:
            caracteristicas['f0_variabilidad'] = np.std(f0_valid)
            caracteristicas['f0_rango'] = np.max(f0_valid) - np.min(f0_valid)
        else:
            caracteristicas['f0_variabilidad'] = 0
            caracteristicas['f0_rango'] = 0
        
        return caracteristicas
        
    except Exception as e:
        print(f"⚠️ Error procesando {path}: {e}")
        return None

# === PROCESAMIENTO DE DATOS ===
print("🔄 Extrayendo características de audio...")
datos = []
total_archivos = 0
archivos_procesados = 0

# Contar total de archivos
for carpeta in carpetas_audio:
    if os.path.exists(carpeta):
        for genero in os.listdir(carpeta):
            path_genero = os.path.join(carpeta, genero)
            if os.path.isdir(path_genero):
                for etiqueta in os.listdir(path_genero):
                    path_etiqueta = os.path.join(path_genero, etiqueta)
                    if os.path.isdir(path_etiqueta):
                        total_archivos += len([f for f in os.listdir(path_etiqueta) if f.endswith(".wav")])

print(f"📊 Total de archivos a procesar: {total_archivos}")

for carpeta in carpetas_audio:
    if not os.path.exists(carpeta):
        print(f"⚠️ Carpeta no encontrada: {carpeta}")
        continue
        
    tipo_audio = "silaba" if "Silaba" in carpeta else "vocal"
    
    for genero in os.listdir(carpeta):
        path_genero = os.path.join(carpeta, genero)
        if not os.path.isdir(path_genero):
            continue

        for etiqueta in os.listdir(path_genero):
            path_etiqueta = os.path.join(path_genero, etiqueta)
            if not os.path.isdir(path_etiqueta):
                continue

            for archivo in os.listdir(path_etiqueta):
                if archivo.endswith(".wav"):
                    ruta = os.path.join(path_etiqueta, archivo)
                    caracteristicas = extraer_caracteristicas_completas(ruta)
                    
                    if caracteristicas is not None:
                        # Clasificación de voz basada en F0
                        f0 = caracteristicas['f0']
                        if f0 < 120:
                            voz = "grave"
                        elif f0 > 200:
                            voz = "aguda"
                        else:
                            voz = "media"
                        
                        # Añadir metadatos
                        caracteristicas.update({
                            "archivo": archivo,
                            "tipo_audio": tipo_audio,
                            "etiqueta": etiqueta,
                            "genero": genero.lower(),
                            "voz": voz
                        })
                        
                        datos.append(caracteristicas)
                        archivos_procesados += 1
                        
                        if archivos_procesados % 10 == 0:
                            print(f"✅ Procesados: {archivos_procesados}/{total_archivos}")

print(f"📁 Archivos procesados exitosamente: {archivos_procesados}/{total_archivos}")

if len(datos) == 0:
    print("❌ No se encontraron datos para entrenar. Verifica la estructura de carpetas.")
    exit()

# === CREAR DATASET ===
df = pd.DataFrame(datos)
df.to_csv(output_csv, index=False)
print(f"💾 Dataset guardado en: {output_csv}")
print(f"📊 Forma del dataset: {df.shape}")
print(f"🏷️ Etiquetas únicas: {df['etiqueta'].unique()}")
print(f"👥 Distribución por género: \n{df['genero'].value_counts()}")

# === PREPARAR CARACTERÍSTICAS PARA ENTRENAMIENTO ===
# Seleccionar características numéricas (excluyendo metadatos)
feature_columns = [col for col in df.columns if col not in 
                  ['archivo', 'tipo_audio', 'etiqueta', 'genero', 'voz']]

X = df[feature_columns]
print(f"🔧 Características utilizadas: {len(feature_columns)}")
print(f"   {feature_columns}")

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

# === ENTRENAMIENTO DE MODELOS ===
print("\n🤖 Entrenando modelos...")

# Configuración mejorada de Random Forest
rf_params = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

resultados_entrenamiento = {}

# Modelo 1: Clasificación de etiquetas (vocal/sílaba específica)
print("🎯 Entrenando modelo de etiquetas...")
encoder = LabelEncoder()
y_etiqueta = encoder.fit_transform(df["etiqueta"])
joblib.dump(encoder, os.path.join(output_dir, "encoder.pkl"))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_etiqueta, 
                                                   test_size=0.2, random_state=42, 
                                                   stratify=y_etiqueta)

modelo_etiquetas = RandomForestClassifier(**rf_params)
modelo_etiquetas.fit(X_train, y_train)

# Evaluación
score_etiquetas = modelo_etiquetas.score(X_test, y_test)
cv_scores = cross_val_score(modelo_etiquetas, X_scaled, y_etiqueta, cv=5)
resultados_entrenamiento['etiquetas'] = {
    'accuracy': score_etiquetas,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}

joblib.dump(modelo_etiquetas, os.path.join(output_dir, "modelo_formantes.pkl"))

# Modelo 2: Género
print("👥 Entrenando modelo de género...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df["genero"], 
                                                   test_size=0.2, random_state=42)

modelo_genero = RandomForestClassifier(**rf_params)
modelo_genero.fit(X_train, y_train)

score_genero = modelo_genero.score(X_test, y_test)
cv_scores = cross_val_score(modelo_genero, X_scaled, df["genero"], cv=5)
resultados_entrenamiento['genero'] = {
    'accuracy': score_genero,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}

joblib.dump(modelo_genero, os.path.join(output_dir, "modelo_genero.pkl"))

# Modelo 3: Tipo de voz
print("🎵 Entrenando modelo de tipo de voz...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df["voz"], 
                                                   test_size=0.2, random_state=42)

modelo_voz = RandomForestClassifier(**rf_params)
modelo_voz.fit(X_train, y_train)

score_voz = modelo_voz.score(X_test, y_test)
cv_scores = cross_val_score(modelo_voz, X_scaled, df["voz"], cv=5)
resultados_entrenamiento['voz'] = {
    'accuracy': score_voz,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}

joblib.dump(modelo_voz, os.path.join(output_dir, "modelo_voz.pkl"))

# Modelo 4: Tipo de audio (vocal vs sílaba)
print("🔤 Entrenando modelo de tipo de audio...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df["tipo_audio"], 
                                                   test_size=0.2, random_state=42)

modelo_tipo = RandomForestClassifier(**rf_params)
modelo_tipo.fit(X_train, y_train)

score_tipo = modelo_tipo.score(X_test, y_test)
cv_scores = cross_val_score(modelo_tipo, X_scaled, df["tipo_audio"], cv=5)
resultados_entrenamiento['tipo_audio'] = {
    'accuracy': score_tipo,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}

joblib.dump(modelo_tipo, os.path.join(output_dir, "modelo_tipo.pkl"))

# === GUARDAR METADATOS ===
metadata = {
    'feature_columns': feature_columns,
    'n_features': len(feature_columns),
    'n_samples': len(df),
    'etiquetas_unicas': df['etiqueta'].unique().tolist(),
    'resultados_entrenamiento': resultados_entrenamiento
}

joblib.dump(metadata, os.path.join(output_dir, "metadata.pkl"))

# === REPORTE FINAL ===
print("\n" + "="*60)
print("📋 REPORTE DE ENTRENAMIENTO")
print("="*60)
print(f"📊 Dataset: {len(df)} muestras, {len(feature_columns)} características")
print(f"🏷️ Etiquetas: {len(df['etiqueta'].unique())} únicas")

for modelo, resultados in resultados_entrenamiento.items():
    print(f"\n🎯 Modelo {modelo}:")
    print(f"   Precisión test: {resultados['accuracy']:.3f}")
    print(f"   CV promedio: {resultados['cv_mean']:.3f} ± {resultados['cv_std']:.3f}")

print(f"\n✅ Modelos guardados en: {output_dir}/")
print("📁 Archivos generados:")
print("   - scaler.pkl (normalizador)")
print("   - encoder.pkl (codificador de etiquetas)")
print("   - modelo_formantes.pkl (clasificación etiquetas)")
print("   - modelo_genero.pkl (clasificación género)")
print("   - modelo_voz.pkl (clasificación tipo voz)")
print("   - modelo_tipo.pkl (vocal vs sílaba)")
print("   - metadata.pkl (metadatos del entrenamiento)")

print("\n🚀 ¡Entrenamiento completado exitosamente!")