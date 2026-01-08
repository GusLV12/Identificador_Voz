# prediccion.py - Versión mejorada compatible con modelos de 20 características
import soundfile as sf
import numpy as np
import scipy.signal
from scipy.signal import find_peaks
import librosa
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

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
            return [700, 1220, 2600, 3400]
        
        # Calcular coeficientes LPC
        a = librosa.lpc(y, order=order)
        
        # Encontrar raíces y convertir a frecuencias
        roots = np.roots(a)
        angles = np.angle(roots)
        freqs = np.abs(angles) * sr / (2 * np.pi)
        
        # Filtrar frecuencias válidas
        valid_freqs = freqs[(freqs > 200) & (freqs < 4000)]
        valid_freqs = np.sort(valid_freqs)
        
        # Asegurar que tenemos los formantes necesarios
        formantes_defaults = [700, 1220, 2600, 3400]
        resultado = []
        
        for i in range(n_formantes):
            if i < len(valid_freqs):
                resultado.append(valid_freqs[i])
            else:
                resultado.append(formantes_defaults[i])
        
        return resultado
        
    except Exception:
        return [700, 1220, 2600, 3400]

def extraer_caracteristicas_completas(path):
    """Extracción completa de 20 características (idéntica al entrenamiento)"""
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
        
        # 2. Formantes reales usando LPC
        formantes = extraer_formantes_lpc(y_trimmed, sr, 4)
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
        print(f"Error procesando {path}: {e}")
        return None

# FUNCIÓN DE COMPATIBILIDAD (mantiene interfaz original)
def extraer_formantes(path):
    """Mantiene compatibilidad con código original pero usa extracción mejorada"""
    caracteristicas = extraer_caracteristicas_completas(path)
    if caracteristicas is None:
        return 150, 700, 1220, 2600
    
    return (
        caracteristicas['f0'],
        caracteristicas['f1'], 
        caracteristicas['f2'],
        caracteristicas['f3']
    )

def predecir_vocal(path):
    """
    FUNCIÓN PRINCIPAL - Mantiene exactamente la misma interfaz
    pero usa análisis mejorado internamente
    """
    if not os.path.exists(path):
        return {
            "tipo": "error",
            "etiqueta": "Archivo no encontrado",
            "genero": "desconocido",
            "voz": "desconocida"
        }
    
    # Extraer todas las características
    caracteristicas = extraer_caracteristicas_completas(path)
    
    if caracteristicas is None:
        return {
            "tipo": "silencio",
            "etiqueta": "Sin audio válido",
            "genero": "desconocido", 
            "voz": "desconocida"
        }
    
    try:
        # Cargar metadatos para saber qué características usar
        metadata = joblib.load("modelos/metadata.pkl")
        feature_columns = metadata['feature_columns']
        
        # Preparar vector de características en el orden correcto
        X = []
        for feature in feature_columns:
            X.append(caracteristicas.get(feature, 0))
        
        X = np.array(X).reshape(1, -1)
        
        # Cargar modelos
        scaler = joblib.load("modelos/scaler.pkl")
        encoder = joblib.load("modelos/encoder.pkl")
        modelo_formantes = joblib.load("modelos/modelo_formantes.pkl")
        modelo_genero = joblib.load("modelos/modelo_genero.pkl")
        modelo_voz = joblib.load("modelos/modelo_voz.pkl")
        
        # Escalar características
        X_scaled = scaler.transform(X)
        
        # Predicciones
        etiqueta = encoder.inverse_transform(modelo_formantes.predict(X_scaled))[0]
        genero = modelo_genero.predict(X_scaled)[0]
        voz = modelo_voz.predict(X_scaled)[0]
        
        # Determinar tipo usando modelo si existe, sino heurística
        if os.path.exists("modelos/modelo_tipo.pkl"):
            modelo_tipo = joblib.load("modelos/modelo_tipo.pkl")
            tipo = modelo_tipo.predict(X_scaled)[0]
        else:
            # Clasificación heurística como fallback
            n_onsets = caracteristicas.get('n_onsets', 0)
            duracion = caracteristicas.get('duracion_efectiva', 0)
            variabilidad = caracteristicas.get('variabilidad_energia', 0)
            
            es_silaba = (n_onsets > 1 or duracion > 0.8 or variabilidad > 0.1)
            tipo = "sílaba" if es_silaba else "vocal"
        
        # Calcular confianza
        probs = modelo_formantes.predict_proba(X_scaled)[0]
        confianza = np.max(probs) * 100
        
        resultado = {
            "tipo": tipo,
            "etiqueta": etiqueta,
            "genero": genero,
            "voz": voz,
            "confianza": round(confianza, 1)
        }
        
        return resultado
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        # Fallback heurístico si falla todo
        return analisis_heuristico_fallback(caracteristicas)

def analisis_heuristico_fallback(caracteristicas):
    """Análisis heurístico si fallan los modelos"""
    f1 = caracteristicas.get('f1', 700)
    f2 = caracteristicas.get('f2', 1220)
    f0 = caracteristicas.get('f0', 150)
    
    # Clasificación básica de vocales
    if f1 < 400 and f2 < 1000:
        vocal = "u"
    elif f1 < 400 and f2 > 2000:
        vocal = "i"
    elif f1 > 700 and f2 > 1500:
        vocal = "a"
    elif f1 < 500 and 1000 < f2 < 1800:
        vocal = "o"
    elif 400 < f1 < 700 and 1200 < f2 < 2000:
        vocal = "e"
    else:
        vocal = "vocal_mixta"
    
    genero = "masculino" if f0 < 165 else "femenino"
    
    if f0 < 120:
        voz = "grave"
    elif f0 > 200:
        voz = "aguda"
    else:
        voz = "media"
    
    n_onsets = caracteristicas.get('n_onsets', 0)
    duracion = caracteristicas.get('duracion_efectiva', 0)
    tipo = "sílaba" if (n_onsets > 1 or duracion > 0.8) else "vocal"
    
    return {
        "tipo": tipo,
        "etiqueta": vocal,
        "genero": genero,
        "voz": voz,
        "confianza": 70.0  # Confianza moderada para heurística
    }