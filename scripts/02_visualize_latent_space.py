#!/usr/bin/env python3
"""
Script para visualizar el espacio latente del VAE usando técnicas de reducción de dimensionalidad.
Genera visualizaciones t-SNE, PCA y UMAP de los embeddings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')

def load_features(parquet_path: str) -> pd.DataFrame:
    """Carga el archivo parquet con los features."""
    print(f"📂 Cargando: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"✅ Shape: {df.shape}")
    return df

def extract_features_and_labels(df: pd.DataFrame):
    """Extrae features numéricas y labels del DataFrame."""
    # Buscar columna de labels
    label_col = None
    for col in ['label', 'class', 'category', 'group', 'filename']:
        if col in df.columns:
            label_col = col
            break
    
    # Extraer features numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].values
    
    # Si hay labels, extraerlas
    labels = None
    if label_col:
        labels = df[label_col].values
        # Si los labels son paths, extraer solo el directorio padre (clase)
        if isinstance(labels[0], str) and '/' in labels[0]:
            labels = np.array([Path(p).parent.name for p in labels])
    
    return X, labels, numeric_cols

def plot_pca(X: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """Visualización PCA del espacio latente."""
    print("\n🔄 Calculando PCA...")
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      c=[colors[idx]], label=label, alpha=0.6, s=50)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50, c='steelblue')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
    ax.set_title('Espacio Latente - PCA', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "04_latent_space_pca.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()
    
    # Varianza explicada
    fig, ax = plt.subplots(figsize=(10, 6))
    n_components = min(20, X.shape[1])
    pca_full = PCA(n_components=n_components)
    pca_full.fit(X_scaled)
    
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    ax.bar(range(1, n_components+1), pca_full.explained_variance_ratio_, 
           alpha=0.7, label='Individual')
    ax.plot(range(1, n_components+1), cumsum, 'ro-', label='Acumulada')
    ax.axhline(y=0.95, color='g', linestyle='--', label='95% varianza')
    ax.set_xlabel('Componente Principal')
    ax.set_ylabel('Varianza Explicada')
    ax.set_title('Varianza Explicada por Componente Principal')
    ax.legend()
    
    output_path = output_dir / "05_pca_variance_explained.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def plot_tsne(X: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """Visualización t-SNE del espacio latente."""
    print("\n🔄 Calculando t-SNE (esto puede tomar un momento)...")
    
    # Submuestrear si hay muchos datos
    n_samples = X.shape[0]
    if n_samples > 5000:
        print(f"   Submuestreando de {n_samples} a 5000 muestras...")
        idx = np.random.choice(n_samples, 5000, replace=False)
        X_subset = X[idx]
        labels_subset = labels[idx] if labels is not None else None
    else:
        X_subset = X
        labels_subset = labels
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if labels_subset is not None:
        unique_labels = np.unique(labels_subset)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for idx, label in enumerate(unique_labels):
            mask = labels_subset == label
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                      c=[colors[idx]], label=label, alpha=0.6, s=50)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=50, c='steelblue')
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Espacio Latente - t-SNE', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "06_latent_space_tsne.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def plot_latent_heatmap(X: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """Heatmap de las features latentes por clase."""
    if labels is None:
        print("⚠️ No hay labels para generar heatmap por clase.")
        return
    
    print("\n🔄 Generando heatmap de features por clase...")
    
    # Calcular media por clase
    unique_labels = np.unique(labels)
    class_means = []
    
    for label in unique_labels:
        mask = labels == label
        class_means.append(X[mask].mean(axis=0))
    
    class_means = np.array(class_means)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Limitar número de features para visualización
    n_features = min(class_means.shape[1], 32)
    
    sns.heatmap(class_means[:, :n_features], 
                xticklabels=[f'z{i}' for i in range(n_features)],
                yticklabels=unique_labels,
                cmap='RdBu_r', center=0, annot=False, ax=ax)
    
    ax.set_xlabel('Dimensión Latente')
    ax.set_ylabel('Clase')
    ax.set_title('Media de Features Latentes por Clase', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "07_latent_heatmap_by_class.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def main():
    # Rutas
    project_root = Path(__file__).parent.parent
    parquet_path = project_root / "embeddings" / "bird_net_vae_audio_splitted" / "features.parquet"
    output_dir = project_root / "outputs" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🌌 VISUALIZADOR DE ESPACIO LATENTE VAE")
    print("="*60)
    
    # Cargar datos
    df = load_features(str(parquet_path))
    
    # Extraer features y labels
    X, labels, feature_cols = extract_features_and_labels(df)
    print(f"📊 Features: {X.shape[1]} dimensiones")
    print(f"📊 Muestras: {X.shape[0]}")
    if labels is not None:
        print(f"📊 Clases: {np.unique(labels)}")
    
    # Generar visualizaciones
    print("\n" + "="*60)
    print("📈 GENERANDO VISUALIZACIONES")
    print("="*60)
    
    plot_pca(X, labels, output_dir)
    plot_tsne(X, labels, output_dir)
    plot_latent_heatmap(X, labels, output_dir)
    
    print("\n" + "="*60)
    print(f"✅ Visualizaciones guardadas en: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
