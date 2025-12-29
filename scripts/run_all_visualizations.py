#!/usr/bin/env python3
"""
Script maestro para ejecutar todas las visualizaciones del proyecto VAE.
Ejecuta todos los scripts de análisis y genera un resumen.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_script(script_path: Path) -> bool:
    """Ejecuta un script Python y retorna si fue exitoso."""
    print(f"\n{'='*60}")
    print(f"🚀 Ejecutando: {script_path.name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            cwd=str(script_path.parent)
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error ejecutando {script_path.name}: {e}")
        return False

def main():
    print("="*60)
    print("🎯 SUITE DE VISUALIZACIÓN VAE")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    scripts_dir = Path(__file__).parent
    
    # Lista de scripts a ejecutar en orden
    scripts = [
        "01_explore_parquet.py",
        "02_visualize_latent_space.py",
        "03_generate_spectrograms.py",
        "04_model_inspection.py",
        "05_vae_reconstruction.py",
    ]
    
    results = {}
    
    for script_name in scripts:
        script_path = scripts_dir / script_name
        if script_path.exists():
            success = run_script(script_path)
            results[script_name] = "✅ OK" if success else "❌ Error"
        else:
            results[script_name] = "⚠️ No encontrado"
    
    # Resumen final
    print("\n" + "="*60)
    print("📊 RESUMEN DE EJECUCIÓN")
    print("="*60)
    
    for script, status in results.items():
        print(f"  {status} {script}")
    
    output_dir = scripts_dir.parent / "outputs" / "visualizations"
    if output_dir.exists():
        n_files = len(list(output_dir.glob("*.png")))
        print(f"\n📁 Total de imágenes generadas: {n_files}")
        print(f"📂 Ubicación: {output_dir}")
    
    print("\n" + "="*60)
    print("✅ Proceso completado")
    print("="*60)

if __name__ == "__main__":
    main()
