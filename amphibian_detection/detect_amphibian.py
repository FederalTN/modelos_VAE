
"""

Toma un audio .wav (externo/val), lo pasa por el encoder del VAE (para obtener z),
y luego lo proyecta a:
- PCA 2D (out-of-sample correcto)
- t-SNE 2D (recalculado incluyendo el punto nuevo; centro/radio SOLO con el set base)

Luego:
- calcula centroide y radio (RR) usando SOLO el set base
- mide distancia del punto nuevo al centroide
- imprime en terminal si está DENTRO o FUERA del radio
- guarda imágenes con centroide (X rojo) y círculo de radio RR + punto nuevo (⭐)
- opcional: guarda los centros/radios en config.json

Requisitos:
pip install librosa numpy pandas matplotlib scikit-learn torch

"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8-darkgrid")


# -------------------------
# Utilidades
# -------------------------
def compute_centroid_and_radius(Z_2d: np.ndarray):
    center = Z_2d.mean(axis=0)
    dists = np.linalg.norm(Z_2d - center, axis=1)
    radius = float(dists.max())
    return center.astype(float), radius


def save_center_radius_to_config(config_path: Path, key: str, center: np.ndarray, radius: float):
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

    # compatibilidad con tu config: último calculado
    data["center"] = {"x": float(center[0]), "y": float(center[1])}
    data["radius"] = float(radius)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def plot_with_center_radius(
    Z_2d: np.ndarray,
    center: np.ndarray,
    radius: float,
    new_point: np.ndarray,
    title: str,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 10))

    # nube base
    ax.scatter(Z_2d[:, 0], Z_2d[:, 1], alpha=0.35, s=35)

    # círculo RR
    circle = plt.Circle((center[0], center[1]), radius, fill=False, linewidth=2)
    ax.add_patch(circle)

    # centroide
    ax.scatter(center[0], center[1], c="red", s=260, marker="X",
               edgecolors="black", linewidths=1.2, label="Centroide")

    # punto nuevo
    ax.scatter(new_point[0], new_point[1], marker="*", s=420,
               edgecolors="black", linewidths=1.0, label="NEW_AUDIO")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Features parquet
# -------------------------
def load_features(parquet_path: Path) -> pd.DataFrame:
    print(f"📂 Cargando parquet: {parquet_path}")
    df = pd.read_parquet(str(parquet_path))
    print(f"✅ Shape: {df.shape}")
    return df


def extract_X(df: pd.DataFrame) -> np.ndarray:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].values.astype(np.float32)
    if X.ndim != 2 or X.shape[1] < 2:
        raise RuntimeError(f"features parquet inválido: X shape={X.shape}")
    return X


# -------------------------
# Audio -> mel -> encoder -> z
# -------------------------
def load_audio_mel(
    audio_path: Path,
    sr: int = 22050,
    duration: float = 3.0,
    n_mels: int = 128,
    fmax: int = 8000,
) -> np.ndarray:
    y, _sr = librosa.load(str(audio_path), sr=sr, duration=duration)
    target_samples = int(duration * sr)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max).astype(np.float32)
    return S_mel_db


def mel_to_tensor(mel_db: np.ndarray, device: str = "cpu") -> torch.Tensor:
    # (n_mels, T) -> (1,1,n_mels,T)
    return torch.from_numpy(mel_db)[None, None, :, :].to(device)


def load_encoder_model(encoder_model_path: Path, device: str = "cpu"):
    # 1) TorchScript
    try:
        model = torch.jit.load(str(encoder_model_path), map_location=device)
        model.eval()
        return model, "torchscript"
    except Exception:
        pass

    # 2) torch.load
    obj = torch.load(str(encoder_model_path), map_location=device)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj, "torch_load_module"

    if isinstance(obj, dict):
        raise RuntimeError(
            "El encoder parece estar guardado como state_dict (dict). "
            "Para usarlo aquí necesitas instanciar la arquitectura del encoder "
            "y hacer model.load_state_dict(state_dict)."
        )

    raise RuntimeError(f"No pude interpretar encoder: {encoder_model_path}")


@torch.no_grad()
def encode_to_z(encoder, x: torch.Tensor) -> np.ndarray:
    out = encoder(x)

    # (mu, logvar)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        mu = out[0]
        return mu.squeeze().detach().cpu().numpy().astype(np.float32)

    # dict con mu
    if isinstance(out, dict):
        for k in ["mu", "mean", "z_mu", "latent_mu"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k].squeeze().detach().cpu().numpy().astype(np.float32)
        for _, v in out.items():
            if torch.is_tensor(v):
                return v.squeeze().detach().cpu().numpy().astype(np.float32)

    # tensor directo
    if torch.is_tensor(out):
        return out.squeeze().detach().cpu().numpy().astype(np.float32)

    raise RuntimeError(f"Salida del encoder no soportada: {type(out)}")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_path", required=True, type=str, help="Ruta al .wav externo/val")
    ap.add_argument("--parquet_path", type=str, default=None, help="Ruta al features.parquet")
    ap.add_argument("--encoder_path", type=str, default=None, help="Ruta al encoder model.pt")
    ap.add_argument("--output_dir", type=str, default=None, help="Directorio de salida de imágenes")
    ap.add_argument("--config_path", type=str, default=None, help="config.json para guardar centros/radios")
    ap.add_argument("--save_config", action="store_true", help="Si se setea, escribe center/radius en config.json")

    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--fmax", type=int, default=8000)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_max_iter", type=int, default=1000)
    ap.add_argument("--tsne_max_base", type=int, default=5000, help="Submuestreo del set base para t-SNE")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    project_root = Path(__file__).parent.parent

    audio_path = Path(args.audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"No existe audio_path: {audio_path}")

    parquet_path = (
        Path(args.parquet_path).expanduser().resolve()
        if args.parquet_path
        else (project_root / "downloaded_models" / "bird_net_vae_audio_splitted_features_v0" / "bird_net_vae_audio_splitted_features.parquet")
    )
    if not parquet_path.exists():
        raise FileNotFoundError(f"No existe parquet_path: {parquet_path}")

    encoder_path = (
        Path(args.encoder_path).expanduser().resolve()
        if args.encoder_path
        else (project_root / "downloaded_models" / "bird_net_vae_audio_splitted_encoder_v0" / "model.pt")
    )
    if not encoder_path.exists():
        raise FileNotFoundError(f"No existe encoder_path: {encoder_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (project_root / "outputs2" / "visualizations")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = (
        Path(args.config_path).expanduser().resolve()
        if args.config_path
        else (project_root / "chunk_maker" / "config.json")
    )

    print("=" * 70)
    print("🧪 CHECK: audio externo vs centroide/radio en PCA y t-SNE")
    print("=" * 70)
    print(f"📌 audio_path:   {audio_path}")
    print(f"📌 parquet_path: {parquet_path}")
    print(f"📌 encoder_path: {encoder_path}")
    print(f"📌 output_dir:   {output_dir}")
    print(f"📌 config_path:  {config_path}")
    print("")

    # 1) Base embeddings (X_base)
    df = load_features(parquet_path)
    X_base = extract_X(df)
    n_base, d = X_base.shape
    print(f"📊 Base: n={n_base}, latent_dim={d}")

    # 2) Encoder + z_new
    encoder, mode = load_encoder_model(encoder_path, device=args.device)
    print(f"✅ Encoder cargado ({mode}).")

    mel_db = load_audio_mel(
        audio_path, sr=args.sr, duration=args.duration, n_mels=args.n_mels, fmax=args.fmax
    )
    x_tensor = mel_to_tensor(mel_db, device=args.device)
    z_new = encode_to_z(encoder, x_tensor).reshape(-1)

    if z_new.shape[0] != d:
        raise RuntimeError(
            f"Dimensión latente no calza: base={d} vs nuevo={z_new.shape[0]}"
        )

    print(f"✅ z_new listo: shape={z_new.shape}")

    # -------------------------
    # PCA: out-of-sample correcto
    # -------------------------
    print("\n🔄 PCA (fit base, transform base+nuevo)...")
    scaler_pca = StandardScaler()
    Xb_scaled = scaler_pca.fit_transform(X_base)
    pca = PCA(n_components=2)
    Zb_pca = pca.fit_transform(Xb_scaled)

    z_new_scaled = scaler_pca.transform(z_new[None, :])
    Znew_pca = pca.transform(z_new_scaled).reshape(-1)

    center_pca, radius_pca = compute_centroid_and_radius(Zb_pca)
    dist_pca = float(np.linalg.norm(Znew_pca - center_pca))
    outside_pca = dist_pca > radius_pca

    comment_pca = "🚨 FUERA" if outside_pca else "✅ DENTRO"
    print(f"{comment_pca} del radio (PCA): dist={dist_pca:.4f} vs RR={radius_pca:.4f}")

    plot_with_center_radius(
        Z_2d=Zb_pca,
        center=center_pca,
        radius=radius_pca,
        new_point=Znew_pca,
        title=f"PCA check — dist={dist_pca:.3f}, RR={radius_pca:.3f} ({comment_pca})",
        out_path=output_dir / "20_check_external_pca_center_radius.png",
    )

    # -------------------------
    # t-SNE: recalculado con punto nuevo (centro/radio solo base)
    # -------------------------
    print("\n🔄 t-SNE (recalculado incluyendo el punto nuevo; centro/radio SOLO base)...")

    # Submuestreo base para t-SNE si es grande
    if n_base > args.tsne_max_base:
        idx = np.random.choice(n_base, args.tsne_max_base, replace=False)
        X_base_tsne = X_base[idx]
        print(f"   Submuestreo base t-SNE: {n_base} -> {X_base_tsne.shape[0]}")
    else:
        idx = None
        X_base_tsne = X_base

    # concateno nuevo al final
    X_tsne_all = np.vstack([X_base_tsne, z_new[None, :]])

    scaler_tsne = StandardScaler()
    Xts_scaled = scaler_tsne.fit_transform(X_tsne_all)

    tsne = TSNE(
        n_components=2,
        perplexity=float(args.tsne_perplexity),
        random_state=int(args.random_state),
        max_iter=int(args.tsne_max_iter),
        init="pca",
        learning_rate="auto",
    )
    Z_all = tsne.fit_transform(Xts_scaled)

    Zb_tsne = Z_all[:-1, :]
    Znew_tsne = Z_all[-1, :].reshape(-1)

    center_tsne, radius_tsne = compute_centroid_and_radius(Zb_tsne)
    dist_tsne = float(np.linalg.norm(Znew_tsne - center_tsne))
    outside_tsne = dist_tsne > radius_tsne

    comment_tsne = "🚨 FUERA" if outside_tsne else "✅ DENTRO"
    print(f"{comment_tsne} del radio (t-SNE): dist={dist_tsne:.4f} vs RR={radius_tsne:.4f}")
    print("   (Nota: t-SNE no es out-of-sample real; esta comparación es práctica/heurística)")

    plot_with_center_radius(
        Z_2d=Zb_tsne,
        center=center_tsne,
        radius=radius_tsne,
        new_point=Znew_tsne,
        title=f"t-SNE check — dist={dist_tsne:.3f}, RR={radius_tsne:.3f} ({comment_tsne})",
        out_path=output_dir / "21_check_external_tsne_center_radius.png",
    )

    # Guardar config si se pide
    if args.save_config:
        save_center_radius_to_config(config_path, "pca", center_pca, radius_pca)
        save_center_radius_to_config(config_path, "tsne", center_tsne, radius_tsne)
        print(f"\n💾 Guardado center/radius en: {config_path}")

    print("\n" + "=" * 70)
    print("✅ Listo.")
    print(f"🖼️ PCA plot:  {output_dir / '20_check_external_pca_center_radius.png'}")
    print(f"🖼️ t-SNE plot:{output_dir / '21_check_external_tsne_center_radius.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
