#!/usr/bin/env python3
"""
Script para inspeccionar los modelos .pt guardados del VAE.
Muestra arquitectura, pesos, estadísticas y visualizaciones de las capas.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import OrderedDict

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')

def load_model(model_path: Path) -> dict:
    """Carga el state dict del modelo."""
    print(f"\n📂 Cargando modelo: {model_path}")
    state_dict = torch.load(str(model_path), map_location='cpu')
    return state_dict

def analyze_state_dict(state_dict: dict, model_name: str) -> dict:
    """Analiza el state dict y extrae información."""
    print(f"\n{'='*60}")
    print(f"📊 ANÁLISIS DEL MODELO: {model_name}")
    print(f"{'='*60}")
    
    total_params = 0
    layer_info = []
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            n_params = param.numel()
            total_params += n_params
            layer_info.append({
                'name': name,
                'shape': tuple(param.shape),
                'params': n_params,
                'dtype': str(param.dtype),
                'mean': float(param.float().mean()),
                'std': float(param.float().std()),
                'min': float(param.float().min()),
                'max': float(param.float().max()),
            })
            print(f"  📌 {name}")
            print(f"     Shape: {tuple(param.shape)} | Params: {n_params:,}")
            print(f"     Mean: {param.float().mean():.4f} | Std: {param.float().std():.4f}")
    
    print(f"\n📊 TOTAL PARÁMETROS: {total_params:,}")
    print(f"📊 TAMAÑO ESTIMADO: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    return {'layers': layer_info, 'total_params': total_params}

def plot_weight_distributions(state_dict: dict, model_name: str, output_dir: Path) -> None:
    """Grafica la distribución de pesos de cada capa."""
    # Filtrar solo tensores de pesos (no biases pequeños)
    weight_layers = [(name, param) for name, param in state_dict.items() 
                     if isinstance(param, torch.Tensor) and param.numel() > 100]
    
    if not weight_layers:
        print("⚠️ No hay capas suficientemente grandes para visualizar.")
        return
    
    n_layers = min(len(weight_layers), 12)  # Máximo 12 capas
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_layers == 1 else axes
    
    for idx, (name, param) in enumerate(weight_layers[:n_layers]):
        ax = axes[idx]
        weights = param.float().flatten().numpy()
        
        ax.hist(weights, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax.set_title(f'{name}\nmean={weights.mean():.3f}, std={weights.std():.3f}', fontsize=9)
        ax.set_xlabel('Valor del peso')
        ax.set_ylabel('Densidad')
    
    # Ocultar ejes vacíos
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Distribución de Pesos - {model_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / f"09_weight_distributions_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def plot_weight_heatmaps(state_dict: dict, model_name: str, output_dir: Path) -> None:
    """Genera heatmaps de las matrices de pesos más interesantes."""
    # Buscar capas lineales o convolucionales con forma 2D
    weight_layers = []
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor) and len(param.shape) == 2:
            if param.shape[0] > 1 and param.shape[1] > 1:
                weight_layers.append((name, param))
    
    if not weight_layers:
        print("⚠️ No hay matrices de pesos 2D para visualizar como heatmap.")
        return
    
    n_layers = min(len(weight_layers), 4)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (name, param) in enumerate(weight_layers[:n_layers]):
        ax = axes[idx]
        weights = param.float().numpy()
        
        # Submuestrear si es muy grande
        if weights.shape[0] > 100:
            weights = weights[::weights.shape[0]//100, :]
        if weights.shape[1] > 100:
            weights = weights[:, ::weights.shape[1]//100]
        
        vmax = max(abs(weights.min()), abs(weights.max()))
        sns.heatmap(weights, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax, 
                    ax=ax, cbar=True, xticklabels=False, yticklabels=False)
        ax.set_title(f'{name}\nShape: {param.shape}', fontsize=10)
    
    # Ocultar ejes vacíos
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Heatmaps de Matrices de Pesos - {model_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / f"10_weight_heatmaps_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def plot_layer_statistics(analysis_results: dict, model_name: str, output_dir: Path) -> None:
    """Grafica estadísticas por capa."""
    layers = analysis_results['layers']
    
    if len(layers) < 2:
        print("⚠️ No hay suficientes capas para graficar estadísticas.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = [l['name'][:30] + '...' if len(l['name']) > 30 else l['name'] for l in layers]
    
    # Número de parámetros por capa
    ax = axes[0, 0]
    params = [l['params'] for l in layers]
    ax.barh(range(len(names)), params, color='steelblue')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Número de Parámetros')
    ax.set_title('Parámetros por Capa')
    ax.invert_yaxis()
    
    # Media de pesos por capa
    ax = axes[0, 1]
    means = [l['mean'] for l in layers]
    colors = ['green' if m >= 0 else 'red' for m in means]
    ax.barh(range(len(names)), means, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Media')
    ax.set_title('Media de Pesos por Capa')
    ax.invert_yaxis()
    
    # Desviación estándar por capa
    ax = axes[1, 0]
    stds = [l['std'] for l in layers]
    ax.barh(range(len(names)), stds, color='orange', alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Desviación Estándar')
    ax.set_title('Std de Pesos por Capa')
    ax.invert_yaxis()
    
    # Rango (max - min) por capa
    ax = axes[1, 1]
    ranges = [l['max'] - l['min'] for l in layers]
    ax.barh(range(len(names)), ranges, color='purple', alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Rango (Max - Min)')
    ax.set_title('Rango de Pesos por Capa')
    ax.invert_yaxis()
    
    fig.suptitle(f'Estadísticas por Capa - {model_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / f"11_layer_statistics_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()

def main():
    # Rutas
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "downloaded_models"
    output_dir = project_root / "outputs" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🔬 INSPECTOR DE MODELOS VAE")
    print("="*60)
    
    # Buscar modelos
    model_paths = list(models_dir.rglob("model.pt"))
    
    if not model_paths:
        print("❌ No se encontraron archivos model.pt")
        return
    
    print(f"\n📁 Modelos encontrados: {len(model_paths)}")
    for path in model_paths:
        print(f"   - {path.relative_to(models_dir)}")
    
    # Analizar cada modelo
    for model_path in model_paths:
        model_name = model_path.parent.name
        
        try:
            state_dict = load_model(model_path)
            
            # Analizar state dict
            analysis = analyze_state_dict(state_dict, model_name)
            
            # Generar visualizaciones
            print(f"\n📈 Generando visualizaciones para {model_name}...")
            plot_weight_distributions(state_dict, model_name, output_dir)
            plot_weight_heatmaps(state_dict, model_name, output_dir)
            plot_layer_statistics(analysis, model_name, output_dir)
            
        except Exception as e:
            print(f"❌ Error procesando {model_name}: {e}")
    
    print("\n" + "="*60)
    print(f"✅ Análisis completado. Resultados en: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
