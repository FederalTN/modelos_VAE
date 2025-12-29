#!/usr/bin/env python3
"""
Script para generar espectrogramas de los audios originales.
Muestra ejemplos de cada clase y compara diferentes representaciones espectrales.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
import random

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')

def get_sample_files(audio_dir: Path, samples_per_class: int = 3) -> dict:
    """Obtiene archivos de audio de muestra de cada clase."""
    samples = {}
    
    for class_dir in audio_dir.iterdir():
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            audio_files = list(class_dir.glob('*.wav')) + list(class_dir.glob('*.mp3'))
            if audio_files:
                n_samples = min(samples_per_class, len(audio_files))
                samples[class_dir.name] = random.sample(audio_files, n_samples)
                print(f"📁 {class_dir.name}: {len(audio_files)} archivos (tomando {n_samples})")
    
    return samples

def plot_waveform_and_spectrogram(audio_path: Path, output_dir: Path, sr: int = 22050) -> None:
    """Genera waveform y espectrograma de un audio."""
    # Cargar audio
    y, sr = librosa.load(str(audio_path), sr=sr)
    
    class_name = audio_path.parent.name
    file_name = audio_path.stem
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Forma de onda
    ax = axes[0]
    librosa.display.waveshow(y, sr=sr, ax=ax, color='steelblue')
    ax.set_title(f'Forma de Onda - {class_name}/{file_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Amplitud')
    
    # 2. Espectrograma Mel
    ax = axes[1]
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    img = librosa.display.specshow(S_mel_db, sr=sr, x_axis='time', y_axis='mel', 
                                    fmax=8000, ax=ax, cmap='magma')
    ax.set_title('Espectrograma Mel', fontsize=12, fontweight='bold')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # 3. Espectrograma STFT
    ax = axes[2]
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
    ax.set_title('Espectrograma STFT', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 8000)  # Limitar a 8kHz
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    plt.tight_layout()
    output_path = output_dir / f"spectrogram_{class_name}_{file_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def plot_class_comparison(samples: dict, output_dir: Path, sr: int = 22050) -> None:
    """Genera una comparación de espectrogramas entre clases."""
    n_classes = len(samples)
    
    fig, axes = plt.subplots(n_classes, 2, figsize=(14, 4*n_classes))
    
    for idx, (class_name, files) in enumerate(samples.items()):
        if not files:
            continue
            
        audio_path = files[0]  # Primer archivo de la clase
        y, sr = librosa.load(str(audio_path), sr=sr)
        
        # Forma de onda
        ax = axes[idx, 0] if n_classes > 1 else axes[0]
        librosa.display.waveshow(y, sr=sr, ax=ax, color='steelblue')
        ax.set_title(f'{class_name} - Forma de Onda', fontsize=11)
        ax.set_xlabel('Tiempo (s)')
        
        # Espectrograma Mel
        ax = axes[idx, 1] if n_classes > 1 else axes[1]
        S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
        librosa.display.specshow(S_mel_db, sr=sr, x_axis='time', y_axis='mel', 
                                  fmax=8000, ax=ax, cmap='magma')
        ax.set_title(f'{class_name} - Espectrograma Mel', fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / "08_class_comparison_spectrograms.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def plot_audio_features(audio_path: Path, output_dir: Path, sr: int = 22050) -> None:
    """Genera gráficos de características de audio adicionales."""
    y, sr = librosa.load(str(audio_path), sr=sr)
    
    class_name = audio_path.parent.name
    file_name = audio_path.stem
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Chromagram
    ax = axes[0, 0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=ax, cmap='coolwarm')
    ax.set_title('Chromagram', fontsize=12, fontweight='bold')
    
    # 2. MFCCs
    ax = axes[0, 1]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax, cmap='coolwarm')
    ax.set_title('MFCCs (20 coeficientes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coeficiente MFCC')
    
    # 3. Spectral Centroid y Bandwidth
    ax = axes[1, 0]
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    times = librosa.times_like(spec_cent, sr=sr)
    ax.plot(times, spec_cent, label='Centroide', color='blue')
    ax.fill_between(times, spec_cent - spec_bw, spec_cent + spec_bw, alpha=0.3, color='blue')
    ax.set_title('Centroide Espectral ± Bandwidth', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Frecuencia (Hz)')
    ax.legend()
    
    # 4. Zero Crossing Rate y RMS
    ax = axes[1, 1]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(zcr, sr=sr)
    ax2 = ax.twinx()
    l1, = ax.plot(times, zcr, label='ZCR', color='green')
    l2, = ax2.plot(times, rms, label='RMS', color='red')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Zero Crossing Rate', color='green')
    ax2.set_ylabel('RMS', color='red')
    ax.set_title('Zero Crossing Rate y RMS', fontsize=12, fontweight='bold')
    ax.legend([l1, l2], ['ZCR', 'RMS'], loc='upper right')
    
    fig.suptitle(f'Características de Audio - {class_name}/{file_name}', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = output_dir / f"audio_features_{class_name}_{file_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def main():
    # Rutas
    project_root = Path(__file__).parent.parent
    audio_dir = project_root / "audio_splitted"
    output_dir = project_root / "outputs" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🎵 GENERADOR DE ESPECTROGRAMAS")
    print("="*60)
    
    # Obtener muestras
    print("\n📂 Buscando archivos de audio...")
    samples = get_sample_files(audio_dir, samples_per_class=2)
    
    if not samples:
        print("❌ No se encontraron archivos de audio.")
        return
    
    # Generar comparación entre clases
    print("\n" + "="*60)
    print("📈 GENERANDO VISUALIZACIONES")
    print("="*60)
    
    plot_class_comparison(samples, output_dir)
    
    # Generar espectrogramas individuales para algunas muestras
    print("\n🎵 Generando espectrogramas individuales...")
    for class_name, files in samples.items():
        if files:
            plot_waveform_and_spectrogram(files[0], output_dir)
            plot_audio_features(files[0], output_dir)
            break  # Solo un ejemplo detallado
    
    print("\n" + "="*60)
    print(f"✅ Espectrogramas guardados en: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
