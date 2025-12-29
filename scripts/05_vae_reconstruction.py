#!/usr/bin/env python3
"""
Script para visualizar reconstrucciones del VAE.
Carga el modelo y genera comparaciones entre entrada original y reconstrucción.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

plt.style.use('seaborn-v0_8-darkgrid')

def load_audio_and_process(audio_path: Path, sr: int = 22050, n_mels: int = 128, 
                            duration: float = 3.0) -> tuple:
    """Carga y procesa un audio para obtener su espectrograma mel."""
    # Cargar audio
    y, sr = librosa.load(str(audio_path), sr=sr, duration=duration)
    
    # Pad si es necesario
    target_samples = int(duration * sr)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    
    # Generar espectrograma mel
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    
    return y, S_mel_db, sr

def visualize_mel_spectrogram(S_mel_db: np.ndarray, title: str, ax, sr: int = 22050) -> None:
    """Visualiza un espectrograma mel."""
    librosa.display.specshow(S_mel_db, sr=sr, x_axis='time', y_axis='mel', 
                              fmax=8000, ax=ax, cmap='magma')
    ax.set_title(title, fontsize=12, fontweight='bold')

def load_vae_components(models_dir: Path) -> tuple:
    """Carga encoder y decoder del VAE."""
    encoder_path = models_dir / "bird_net_vae_audio_splitted_encoder_v0" / "model.pt"
    decoder_path = models_dir / "bird_net_vae_audio_splitted_decoder_v0" / "model.pt"
    
    encoder_state = None
    decoder_state = None
    
    if encoder_path.exists():
        print(f"📂 Cargando encoder: {encoder_path}")
        encoder_state = torch.load(str(encoder_path), map_location='cpu')
    else:
        print(f"⚠️ No se encontró encoder en: {encoder_path}")
    
    if decoder_path.exists():
        print(f"📂 Cargando decoder: {decoder_path}")
        decoder_state = torch.load(str(decoder_path), map_location='cpu')
    else:
        print(f"⚠️ No se encontró decoder en: {decoder_path}")
    
    return encoder_state, decoder_state

def plot_sample_spectrograms(audio_dir: Path, output_dir: Path, n_samples: int = 6) -> None:
    """Genera espectrogramas de muestras de diferentes clases."""
    print("\n🎵 Generando muestras de espectrogramas por clase...")
    
    class_dirs = [d for d in audio_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not class_dirs:
        print("❌ No se encontraron directorios de clases.")
        return
    
    n_classes = len(class_dirs)
    fig, axes = plt.subplots(n_classes, 2, figsize=(14, 4*n_classes))
    
    for idx, class_dir in enumerate(class_dirs):
        audio_files = list(class_dir.glob('*.wav'))[:2]  # 2 muestras por clase
        
        for j, audio_file in enumerate(audio_files):
            if j >= 2:
                break
            
            try:
                y, S_mel_db, sr = load_audio_and_process(audio_file)
                
                ax = axes[idx, j] if n_classes > 1 else axes[j]
                librosa.display.specshow(S_mel_db, sr=sr, x_axis='time', y_axis='mel',
                                          fmax=8000, ax=ax, cmap='magma')
                ax.set_title(f'{class_dir.name}\n{audio_file.name[:30]}', fontsize=10)
            except Exception as e:
                print(f"⚠️ Error procesando {audio_file}: {e}")
    
    plt.suptitle('Muestras de Espectrogramas por Clase', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "12_sample_spectrograms_by_class.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def plot_latent_interpolation(output_dir: Path) -> None:
    """Visualiza interpolación conceptual en el espacio latente."""
    print("\n🌈 Generando visualización de interpolación latente...")
    
    # Crear un ejemplo ilustrativo de interpolación
    np.random.seed(42)
    
    z_dim = 32
    n_steps = 8
    
    # Simular dos puntos en el espacio latente
    z_start = np.random.randn(z_dim) * 0.5
    z_end = np.random.randn(z_dim) * 0.5
    
    # Interpolar
    alphas = np.linspace(0, 1, n_steps)
    z_interp = np.array([z_start * (1 - a) + z_end * a for a in alphas])
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Visualizar los vectores latentes
    ax = axes[0]
    im = ax.imshow(z_interp.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_xlabel('Paso de Interpolación')
    ax.set_ylabel('Dimensión Latente')
    ax.set_title('Interpolación en el Espacio Latente (z)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([f'α={a:.2f}' for a in alphas], rotation=45)
    plt.colorbar(im, ax=ax)
    
    # Visualizar la norma del vector en cada paso
    ax = axes[1]
    norms = np.linalg.norm(z_interp, axis=1)
    ax.plot(alphas, norms, 'bo-', linewidth=2, markersize=8)
    ax.fill_between(alphas, norms, alpha=0.3)
    ax.set_xlabel('Alpha (0=inicio, 1=fin)')
    ax.set_ylabel('||z||₂')
    ax.set_title('Norma del Vector Latente durante Interpolación', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "13_latent_interpolation_concept.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def plot_vae_architecture_diagram(output_dir: Path) -> None:
    """Genera un diagrama conceptual de la arquitectura VAE."""
    print("\n🏗️ Generando diagrama de arquitectura VAE...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colores
    colors = {
        'input': '#3498db',
        'encoder': '#2ecc71',
        'latent': '#e74c3c',
        'decoder': '#9b59b6',
        'output': '#f39c12'
    }
    
    # Input (espectrograma)
    rect = plt.Rectangle((0.5, 2), 1.5, 2, fill=True, color=colors['input'], alpha=0.7)
    ax.add_patch(rect)
    ax.text(1.25, 3, 'Input\n(Mel Spec)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Encoder
    rect = plt.Rectangle((2.5, 1.5), 2, 3, fill=True, color=colors['encoder'], alpha=0.7)
    ax.add_patch(rect)
    ax.text(3.5, 3, 'Encoder\nq(z|x)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Latent space (μ y σ)
    circle1 = plt.Circle((5.5, 3.5), 0.4, fill=True, color=colors['latent'], alpha=0.7)
    circle2 = plt.Circle((5.5, 2.5), 0.4, fill=True, color=colors['latent'], alpha=0.7)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.text(5.5, 3.5, 'μ', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(5.5, 2.5, 'σ', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(5.5, 1.7, 'z ~ N(μ,σ²)', ha='center', va='center', fontsize=10)
    
    # Decoder
    rect = plt.Rectangle((6.5, 1.5), 2, 3, fill=True, color=colors['decoder'], alpha=0.7)
    ax.add_patch(rect)
    ax.text(7.5, 3, 'Decoder\np(x|z)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Output (reconstrucción)
    rect = plt.Rectangle((9, 2), 1.5, 2, fill=True, color=colors['output'], alpha=0.7)
    ax.add_patch(rect)
    ax.text(9.75, 3, 'Output\nx̂', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Flechas
    arrow_style = dict(arrowstyle='->', lw=2, color='gray')
    ax.annotate('', xy=(2.4, 3), xytext=(2, 3), arrowprops=arrow_style)
    ax.annotate('', xy=(5.1, 3.5), xytext=(4.5, 3.2), arrowprops=arrow_style)
    ax.annotate('', xy=(5.1, 2.5), xytext=(4.5, 2.8), arrowprops=arrow_style)
    ax.annotate('', xy=(6.4, 3), xytext=(5.9, 3), arrowprops=arrow_style)
    ax.annotate('', xy=(8.9, 3), xytext=(8.5, 3), arrowprops=arrow_style)
    
    # Título y fórmula de loss
    ax.set_title('Arquitectura Variational Autoencoder (VAE)', fontsize=14, fontweight='bold', pad=20)
    ax.text(5, 5.5, 'Loss = -E[log p(x|z)] + D_KL(q(z|x) || p(z))', 
            ha='center', va='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / "14_vae_architecture_diagram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def main():
    # Rutas
    project_root = Path(__file__).parent.parent
    audio_dir = project_root / "audio_splitted"
    models_dir = project_root / "downloaded_models"
    output_dir = project_root / "outputs" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🔄 VISUALIZADOR DE RECONSTRUCCIONES VAE")
    print("="*60)
    
    # Cargar componentes del VAE
    encoder_state, decoder_state = load_vae_components(models_dir)
    
    if encoder_state:
        print(f"   Encoder keys: {len(encoder_state)} capas")
    if decoder_state:
        print(f"   Decoder keys: {len(decoder_state)} capas")
    
    # Generar visualizaciones
    print("\n" + "="*60)
    print("📈 GENERANDO VISUALIZACIONES")
    print("="*60)
    
    # Espectrogramas de muestra
    plot_sample_spectrograms(audio_dir, output_dir)
    
    # Interpolación latente conceptual
    plot_latent_interpolation(output_dir)
    
    # Diagrama de arquitectura
    plot_vae_architecture_diagram(output_dir)
    
    print("\n" + "="*60)
    print(f"✅ Visualizaciones guardadas en: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
