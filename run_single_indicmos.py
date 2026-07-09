import argparse
import array
import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
INDICMOS_DIR = ROOT / "IndicMOS"
if str(INDICMOS_DIR) not in sys.path:
    sys.path.insert(0, str(INDICMOS_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# infer_indicmos.py prints debug info at import time in this checkout.
with contextlib.redirect_stdout(io.StringIO()):
    from infer_indicmos import LANG_ID_MAPPING, HF_PATH, load_model


def load_audio_16k_mono(audio_path: Path) -> torch.Tensor:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    audio = array.array("f")
    audio.frombytes(result.stdout)
    if len(audio) == 0:
        raise ValueError(f"No audio samples decoded from {audio_path}")
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)


def predict_mos(
    audio_path: Path,
    langid: str | None = None,
    use_langid: bool = False,
    device: str = "cpu",
    model_cache: str = HF_PATH,
) -> float:
    if use_langid and not langid:
        raise ValueError("--langid is required when --use-langid is set")

    wav = load_audio_16k_mono(audio_path).to(device)
    lang_tensor = None

    if use_langid:
        langid = langid.lower()
        if langid not in LANG_ID_MAPPING:
            supported = ", ".join(sorted(LANG_ID_MAPPING))
            raise ValueError(f"Unsupported langid '{langid}'. Use one of: {supported}")
        lang_tensor = torch.tensor([LANG_ID_MAPPING[langid]], dtype=torch.long, device=device)

    model = load_model(
        use_cer=False,
        use_langid=use_langid,
        download_path=model_cache,
        device=device,
    )

    with torch.no_grad():
        lengths = torch.tensor([wav.shape[-1]], dtype=torch.long, device=device)
        score = model(
            wav,
            lang_data=lang_tensor,
            lengths=lengths,
            batch_mode=True,
        ).squeeze().cpu().item()

    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IndicMOS on one audio file.")
    parser.add_argument("audio_path", type=Path, help="Path to one WAV/MP3/FLAC audio file")
    parser.add_argument("--langid", default=None, help="Optional language id, e.g. hi, te, mr, kn, bn, en")
    parser.add_argument("--use-langid", action="store_true", help="Use IndicMOS language-ID model")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--model-cache", default=HF_PATH, help="Hugging Face model cache folder")
    args = parser.parse_args()

    if not args.audio_path.exists():
        raise FileNotFoundError(args.audio_path)

    mos = predict_mos(
        audio_path=args.audio_path,
        langid=args.langid,
        use_langid=args.use_langid,
        device=args.device,
        model_cache=args.model_cache,
    )
    print(f"IndicMOS: {mos:.4f}")


if __name__ == "__main__":
    main()
