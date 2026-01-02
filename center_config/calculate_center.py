#!/usr/bin/env python3
"""
Script para visualizar el espacio latente del VAE usando técnicas de reducción de dimensionalidad.
Genera visualizaciones t-SNE, PCA y (opcional) UMAP de los embeddings.

Además:
- Calcula el centroide (CRx, CRy) y el radio (RR = distancia máxima al punto más alejado) en PCA 2D y t-SNE 2D.
- Genera imágenes extra con el centroide marcado en rojo.
- Guarda centroide y radio en config.json.
"""

import json
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
        if len(labels) > 0 and isinstance(labels[0], str) and ('/' in labels[0] or '\\' in labels[0]):
            labels = np.array([Path(p).parent.name for p in labels])

    return X, labels, numeric_cols


def compute_centroid_and_radius(Z_2d: np.ndarray):
    """
    Calcula el centroide (CRx, CRy) y el radio RR (distancia máxima al centroide)
    en un embedding 2D.
    """
    center = Z_2d.mean(axis=0)  # [CRx, CRy]
    dists = np.linalg.norm(Z_2d - center, axis=1)
    radius = float(dists.max())
    return center.astype(float), radius


def save_center_radius_to_config(config_path: Path, key: str, center: np.ndarray, radius: float):
    """
    Guarda info en config.json sin romper campos existentes.
    - Mantiene root_dir, chunk_seconds, output_dir, center, radius
    - Añade center_pca / radius_pca y/o center_tsne / radius_tsne
    """
    data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if key == "pca":
        data["center_pca"] = {"x": float(center[0]), "y": float(center[1])}
        data["radius_pca"] = float(radius)
    elif key == "tsne":
        data["center_tsne"] = {"x": float(center[0]), "y": float(center[1])}
        data["radius_tsne"] = float(radius)

    # Mantener compatibilidad con tu config original:
    # 'center' y 'radius' apuntan al último calculado.
    data["center"] = {"x": float(center[0]), "y": float(center[1])}
    data["radius"] = float(radius)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def plot_pca(X: np.ndarray, labels: np.ndarray, output_dir: Path, config_path: Path) -> None:
    """Visualización PCA del espacio latente + centroide/radio."""
    print("\n🔄 Calculando PCA...")

    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Centroide y radio en PCA 2D
    center_pca, radius_pca = compute_centroid_and_radius(X_pca)

    # Plot PCA (normal)
    fig, ax = plt.subplots(figsize=(12, 10))

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for idx, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[colors[idx]], label=str(label), alpha=0.6, s=50)
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

    # Imagen extra PCA + centroide (punto rojo)
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.35, s=40)
    ax2.scatter(center_pca[0], center_pca[1], c="red", s=250, marker="X",
                edgecolors="black", linewidths=1.2)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
    ax2.set_title(f'PCA + Centroide (RR={radius_pca:.3f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path2 = output_dir / "08_latent_space_pca_centroid.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path2}")
    plt.close()

    # Varianza explicada
    fig, ax = plt.subplots(figsize=(10, 6))
    n_components = min(20, X.shape[1])
    pca_full = PCA(n_components=n_components)
    pca_full.fit(X_scaled)

    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    ax.bar(range(1, n_components + 1), pca_full.explained_variance_ratio_,
           alpha=0.7, label='Individual')
    ax.plot(range(1, n_components + 1), cumsum, 'ro-', label='Acumulada')
    ax.axhline(y=0.95, color='g', linestyle='--', label='95% varianza')
    ax.set_xlabel('Componente Principal')
    ax.set_ylabel('Varianza Explicada')
    ax.set_title('Varianza Explicada por Componente Principal')
    ax.legend()

    output_path = output_dir / "05_pca_variance_explained.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

    # Guardar centroide/radio PCA en config.json
    save_center_radius_to_config(config_path, "pca", center_pca, radius_pca)
    print(f"✅ Centroide/radio PCA guardados en: {config_path}")


def plot_tsne(X: np.ndarray, labels: np.ndarray, output_dir: Path, config_path: Path) -> None:
    """Visualización t-SNE del espacio latente + centroide/radio."""
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

    # t-SNE (sklearn actual usa max_iter)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        max_iter=1000,
        init="pca",
        learning_rate="auto"
    )
    X_tsne = tsne.fit_transform(X_scaled)

    # Centroide y radio en t-SNE 2D
    center_tsne, radius_tsne = compute_centroid_and_radius(X_tsne)

    fig, ax = plt.subplots(figsize=(12, 10))

    if labels_subset is not None:
        unique_labels = np.unique(labels_subset)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for idx, label in enumerate(unique_labels):
            mask = labels_subset == label
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=[colors[idx]], label=str(label), alpha=0.6, s=50)
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

    # Imagen extra t-SNE + centroide (punto rojo)
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.35, s=40)
    ax2.scatter(center_tsne[0], center_tsne[1], c="red", s=250, marker="X",
                edgecolors="black", linewidths=1.2)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title(f't-SNE + Centroide (RR={radius_tsne:.3f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path2 = output_dir / "09_latent_space_tsne_centroid.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path2}")
    plt.close()

    # Guardar centroide/radio t-SNE en config.json
    save_center_radius_to_config(config_path, "tsne", center_tsne, radius_tsne)
    print(f"✅ Centroide/radio t-SNE guardados en: {config_path}")


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

    sns.heatmap(
        class_means[:, :n_features],
        xticklabels=[f'z{i}' for i in range(n_features)],
        yticklabels=unique_labels,
        cmap='RdBu_r',
        center=0,
        annot=False,
        ax=ax
    )

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
    parquet_path = project_root / "downloaded_models" / "bird_net_vae_audio_splitted_features_v0" / "bird_net_vae_audio_splitted_features.parquet"

    output_dir = project_root / "outputs2" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # config.json en la raíz del proyecto
    config_path = project_root / "chunk_maker" / "config.json"

    print("=" * 60)
    print("🌌 VISUALIZADOR DE ESPACIO LATENTE VAE")
    print("=" * 60)

    # Cargar datos
    df = load_features(str(parquet_path))

    # Extraer features y labels
    X, labels, feature_cols = extract_features_and_labels(df)
    print(f"📊 Features: {X.shape[1]} dimensiones")
    print(f"📊 Muestras: {X.shape[0]}")
    if labels is not None:
        print(f"📊 Clases: {np.unique(labels)}")

    # Generar visualizaciones
    print("\n" + "=" * 60)
    print("📈 GENERANDO VISUALIZACIONES")
    print("=" * 60)

    plot_pca(X, labels, output_dir, config_path)
    plot_tsne(X, labels, output_dir, config_path)
    plot_latent_heatmap(X, labels, output_dir)

    print("\n" + "=" * 60)
    print(f"✅ Visualizaciones guardadas en: {output_dir}")
    print(f"✅ Config actualizado en: {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
