import json
import math
from pathlib import Path

import soundfile as sf


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "root_dir" not in cfg or "chunk_seconds" not in cfg:
        raise ValueError("config.json debe incluir 'root_dir' y 'chunk_seconds'.")

    cfg["root_dir"] = Path(cfg["root_dir"])
    cfg["output_dir"] = Path(cfg.get("output_dir", str(cfg["root_dir"] / "chunks_output")))
    cfg["chunk_seconds"] = float(cfg["chunk_seconds"])

    if cfg["chunk_seconds"] <= 0:
        raise ValueError("'chunk_seconds' debe ser > 0.")

    return cfg


def chunk_wav_file(wav_path: Path, chunk_seconds: float, out_dir: Path) -> tuple[int, int]:
    """
    Devuelve (chunks_creados, descartado_flag)
    descartado_flag: 1 si el archivo completo se descartó por durar menos de chunk_seconds, si no 0.
    """
    info = sf.info(str(wav_path))
    sr = info.samplerate
    total_frames = info.frames

    chunk_frames = int(round(chunk_seconds * sr))
    if chunk_frames <= 0:
        raise ValueError("chunk_frames calculado <= 0. Revisa chunk_seconds.")

    # Si el audio completo dura menos de x, descartamos.
    if total_frames < chunk_frames:
        return 0, 1

    # Número de chunks completos de tamaño exacto x
    full_chunks = total_frames // chunk_frames
    if full_chunks <= 0:
        return 0, 1

    # Leer y escribir por ventanas para no cargar todo si es grande
    created = 0
    base_name = wav_path.stem

    out_dir.mkdir(parents=True, exist_ok=True)

    with sf.SoundFile(str(wav_path), mode="r") as f:
        channels = f.channels
        subtype = info.subtype  # conserva formato (PCM_16, etc.)

        for i in range(full_chunks):
            start = i * chunk_frames
            f.seek(start)
            data = f.read(chunk_frames, dtype="float32", always_2d=True)

            # Protección: si por cualquier motivo no leyó exactamente chunk_frames, descartamos ese trozo
            if data.shape[0] < chunk_frames:
                break

            # Si el original era mono, guardamos mono (no 2D)
            if channels == 1:
                data_to_write = data[:, 0]
            else:
                data_to_write = data

            out_name = f"{base_name}_chunk{i+1}.wav"
            out_path = out_dir / out_name
            sf.write(str(out_path), data_to_write, sr, subtype=subtype)
            created += 1

    return created, 0


def main():
    config_path = Path(__file__).with_name("config.json")
    cfg = load_config(config_path)

    root_dir: Path = cfg["root_dir"]
    out_root: Path = cfg["output_dir"]
    chunk_seconds: float = cfg["chunk_seconds"]

    if not root_dir.exists():
        raise FileNotFoundError(f"No existe root_dir: {root_dir}")

    total_files = 0
    total_discarded = 0
    total_chunks = 0

    # Recorre carpetas dentro de root_dir (y opcionalmente sub-subcarpetas)
    # Si solo quieres 1 nivel (solo esas carpetas de especies), cambia rglob por glob("*.wav") dentro de cada subcarpeta.
    for wav_path in root_dir.rglob("*.wav"):
        total_files += 1

        # Replica estructura de carpetas en output:
        # output_dir/<ruta_relativa_de_la_carpeta_del_wav>/
        rel_parent = wav_path.parent.relative_to(root_dir)
        out_dir = out_root / rel_parent

        created, discarded = chunk_wav_file(wav_path, chunk_seconds, out_dir)
        total_chunks += created
        total_discarded += discarded

    print("=== RESUMEN ===")
    print(f"Root:           {root_dir}")
    print(f"Output:         {out_root}")
    print(f"Chunk seconds:  {chunk_seconds}")
    print(f"WAVs encontrados:     {total_files}")
    print(f"WAVs descartados (<x): {total_discarded}")
    print(f"Chunks creados:        {total_chunks}")


if __name__ == "__main__":
    main()
