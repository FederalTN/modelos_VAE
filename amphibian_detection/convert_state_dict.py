"""Utility: convert a PyTorch state_dict to a TorchScript module.

Usage example:
  python3 convert_state_dict.py \
    --state_path downloaded_models/bird_net_vae_audio_splitted_encoder_v0/model.pt \
    --encoder_arch "my_package.my_module:Encoder" \
    --init_args '{"in_ch":1, "latent_dim":257}' \
    --output_path downloaded_models/bird_net_vae_audio_splitted_encoder_v0/encoder_ts.pt

The script will import `my_package.my_module`, instantiate `Encoder(**init_args)`,
load the state_dict, convert to TorchScript with `torch.jit.script`, and save the traced module.

If the class can be instantiated without args, omit `--init_args`.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_path", required=True, help="Path to the state_dict .pt file")
    ap.add_argument(
        "--encoder_arch",
        required=True,
        help="Import path to encoder class in format 'module.path:ClassName'",
    )
    ap.add_argument(
        "--init_args",
        type=str,
        default=None,
        help="JSON string of kwargs to pass to the class constructor, e.g. '{\"in_ch\":1,\"latent_dim\":257}'",
    )
    ap.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the TorchScript file. If omitted, saves next to state_path with suffix '_ts.pt'",
    )
    args = ap.parse_args()

    state_path = Path(args.state_path).expanduser().resolve()
    if not state_path.exists():
        print(f"State dict not found: {state_path}")
        sys.exit(2)

    try:
        module_name, class_name = args.encoder_arch.split(":")
    except Exception:
        print("--encoder_arch must be 'module.path:ClassName'")
        sys.exit(2)

    try:
        module = importlib.import_module(module_name)
    except Exception as ex:
        print(f"Could not import module '{module_name}': {ex}")
        sys.exit(2)

    if not hasattr(module, class_name):
        print(f"Module '{module_name}' does not contain class '{class_name}'")
        sys.exit(2)

    EncoderClass = getattr(module, class_name)

    init_kwargs = {}
    if args.init_args:
        try:
            init_kwargs = json.loads(args.init_args)
        except Exception as ex:
            print(f"Could not parse --init_args JSON: {ex}")
            sys.exit(2)

    try:
        model = EncoderClass(**init_kwargs)
    except TypeError as ex:
        print(f"Failed to instantiate {class_name} with provided init args: {ex}")
        sys.exit(2)

    sd = torch.load(str(state_path), map_location="cpu")
    if not isinstance(sd, dict):
        print("The provided state_path does not look like a state_dict (dict). Aborting.")
        sys.exit(2)

    try:
        model.load_state_dict(sd)
    except Exception as ex:
        print(f"load_state_dict failed: {ex}")
        sys.exit(2)

    model.eval()

    try:
        scripted = torch.jit.script(model)
    except Exception as ex:
        print(f"torch.jit.script failed: {ex}\nTrying torch.jit.trace with a dummy input...")
        # Try trace: produce a plausible dummy input (best-effort)
        # We attempt common shapes used in this repo: (1,1,128,~130)
        try:
            dummy = torch.zeros(1, 1, 128, 130)
            scripted = torch.jit.trace(model, dummy)
        except Exception as ex2:
            print(f"torch.jit.trace also failed: {ex2}")
            sys.exit(2)

    out_path = Path(args.output_path) if args.output_path else state_path.with_name(state_path.stem + "_ts.pt")
    torch.jit.save(scripted, str(out_path))
    print(f"Saved TorchScript encoder to: {out_path}")


if __name__ == "__main__":
    main()
