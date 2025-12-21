import json, torch, torchaudio, noisereduce as nr, whisperx
from pathlib import Path
from datetime import datetime
from whisperx.diarize import DiarizationPipeline

HF_TOKEN = "YOUR_HF_TOKEN"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "Data" / "voice" / "meeting voice datas"
OUT_DIR = PROJECT_ROOT / "Data" / "voice" / "transcripts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_voice_stt(build_mode=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    diarizer = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        device=device,
        use_auth_token=HF_TOKEN
    )

    for audio_path in AUDIO_DIR.glob("*.*"):
        if audio_path.suffix.lower() not in {".wav", ".mp3"}:
            continue

        out_json = OUT_DIR / f"{audio_path.stem}.json"
        if build_mode and out_json.exists():
            print(f"⏭ 이미 처리됨 → {out_json.name}")
            continue

        # 1️⃣ Load & denoise
        wav, sr = torchaudio.load(str(audio_path))
        wav = wav.mean(dim=0)  # mono
        noise = wav[:sr]
        clean = nr.reduce_noise(y=wav.numpy(), sr=sr, y_noise=noise)

        clean_path = OUT_DIR / f"{audio_path.stem}_clean.wav"
        torchaudio.save(
            str(clean_path),
            torch.tensor(clean).unsqueeze(0),
            sr
        )

        # 2️⃣ WhisperX STT
        model = whisperx.load_model(
            "medium",
            device=device,
            compute_type="int8"
        )

        audio_array = whisperx.load_audio(str(clean_path))
        result = model.transcribe(audio_array, batch_size=16)

        # 3️⃣ Diarization
        diarization = diarizer(audio_array)
        final = whisperx.assign_word_speakers(diarization, result)

        # 4️⃣ Save JSON
        doc = {
            "doc_id": audio_path.stem,   # ✅ Path 객체
            "created_at": datetime.now().isoformat(),
            "segments": final["segments"]
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        print(f"✅ Voice STT 완료 → {out_json.name}")
