#!/usr/bin/env python3
"""
Script para explorar y visualizar el contenido del archivo parquet con los features extraídos del VAE.
Genera estadísticas, distribuciones y gráficos de los embeddings latentes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_features(parquet_path: str) -> pd.DataFrame:
    """Carga el archivo parquet con los features."""
    print(f"📂 Cargando archivo: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"✅ Cargado correctamente!")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columnas: {list(df.columns)}")
    print(f"   - Tipos de datos:\n{df.dtypes}")
    return df

def show_basic_stats(df: pd.DataFrame) -> None:
    """Muestra estadísticas básicas del DataFrame."""
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS BÁSICAS")
    print("="*60)
    print(f"\nDescripción estadística:")
    print(df.describe())
    
    # Si hay columna de clase/label
    for col in ['label', 'class', 'category', 'group']:
        if col in df.columns:
            print(f"\n📌 Distribución por '{col}':")
            print(df[col].value_counts())

def plot_feature_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Grafica la distribución de las features numéricas."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("⚠️ No se encontraron columnas numéricas para graficar.")
        return
    
    n_cols = min(len(numeric_cols), 16)  # Máximo 16 features
    n_rows = (n_cols + 3) // 4
    
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes
    
    for idx, col in enumerate(numeric_cols[:16]):
        ax = axes[idx]
        ax.hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'Distribución: {col}', fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel('Frecuencia')
    
    # Ocultar ejes vacíos
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / "01_feature_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    """Grafica la matriz de correlación de features numéricas."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        print("⚠️ No hay suficientes columnas numéricas para matriz de correlación.")
        return
    
    # Limitar a 20 columnas para visualización clara
    if numeric_df.shape[1] > 20:
        numeric_df = numeric_df.iloc[:, :20]
    
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Matriz de Correlación de Features Latentes', fontsize=14)
    
    plt.tight_layout()
    output_path = output_dir / "02_correlation_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def plot_pairplot(df: pd.DataFrame, output_dir: Path, label_col: str = None) -> None:
    """Genera un scatter plot matrix de las primeras features."""
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns[:4])  # Solo 4 features
    
    if len(numeric_cols) < 2:
        print("⚠️ No hay suficientes columnas para pairplot.")
        return
    
    n = len(numeric_cols)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(df[numeric_cols[i]].dropna(), bins=30, alpha=0.7, edgecolor='black')
            else:
                ax.scatter(df[numeric_cols[j]], df[numeric_cols[i]], alpha=0.3, s=10)
            
            if i == n-1:
                ax.set_xlabel(numeric_cols[j][:15], fontsize=8)
            if j == 0:
                ax.set_ylabel(numeric_cols[i][:15], fontsize=8)
    
    fig.suptitle('Scatter Plot Matrix de Features Latentes (primeras 4)', y=1.02, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "03_pairplot.png"
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
    print("🔍 EXPLORADOR DE FEATURES DEL VAE")
    print("="*60)
    
    # Cargar datos
    df = load_features(str(parquet_path))
    
    # Mostrar estadísticas
    show_basic_stats(df)
    
    # Generar visualizaciones
    print("\n" + "="*60)
    print("📈 GENERANDO VISUALIZACIONES")
    print("="*60)
    
    plot_feature_distributions(df, output_dir)
    plot_correlation_matrix(df, output_dir)
    plot_pairplot(df, output_dir)
    
    print("\n" + "="*60)
    print(f"✅ Todas las visualizaciones guardadas en: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
