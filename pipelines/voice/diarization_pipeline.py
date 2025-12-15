import json
import torch
import torchaudio
import noisereduce as nr
import whisperx
from pathlib import Path
from datetime import datetime
import os
from whisperx.diarize import DiarizationPipeline

# ======================
# 설정
# ======================
HF_TOKEN = '' # 허깅 페이스 토큰 다시 입력하기, hf_로 시작하는 문자열
AUDIO_DIR = Path(r"C:\Users\proto\Desktop\Nagyeong\Global_Capstone\Data\voice\meeting voice datas")
OUTPUT_DIR = Path("data/voice")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# 1. 노이즈 제거
# ======================
def denoise_audio(path: Path) -> Path:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)  # mono

    noise_sample = wav[:sr]
    wav_denoised = nr.reduce_noise(y=wav.numpy(), sr=sr, y_noise=noise_sample)

    out = OUTPUT_DIR / f"{path.stem}_clean.wav"
    torchaudio.save(str(out), torch.tensor(wav_denoised).unsqueeze(0), sr)
    print(f"🧹 노이즈 제거 완료 → {out}")
    return out


# ======================
# 2. WhisperX 기반 STT + 화자 분리
# ======================
def whisperx_diarization(clean_wav: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🎙 WhisperX 모델 로딩 중... ({clean_wav.stem})")
    model = whisperx.load_model("medium", device=device, compute_type="int8")

    audio = whisperx.load_audio(str(clean_wav))
    stt_result = model.transcribe(audio, batch_size=16)

    print("🧠 WhisperX alignment 모델 로딩...")
    align_model, metadata = whisperx.load_align_model(
        language_code=stt_result["language"], device=device
    )
    aligned = whisperx.align(
        stt_result["segments"], align_model, metadata, audio, device=device
    )

    print("🗣 WhisperX diarization 모델 로딩...")

    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        device=device,
        use_auth_token=HF_TOKEN
    )

    diarization_segments = diarize_model(audio)

    print("🔗 화자 + STT 결합...")
    final_segments = whisperx.assign_word_speakers(
        diarization_segments,
        stt_result
    )

    return final_segments


# ======================
# 3. JSON 회의록 저장
# ======================
def save_json(final_segments, source_file: Path):
    out_path = OUTPUT_DIR / f"{source_file.stem}.json"

    doc = {
        "doc_id": source_file.stem,
        "source": "voice",
        "created_at": datetime.now().isoformat(),
        "segments": []
    }

    for seg in final_segments["segments"]:
        doc["segments"].append({
            "speaker": seg.get("speaker", "Unknown"),
            "text": seg.get("text", ""),
            "start": seg.get("start", 0),
            "end": seg.get("end", 0)
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"💾 저장됨 → {out_path}")
    return out_path


# ======================
# 4. 여러 파일 처리 + build 모드
# ======================
def process_all_audio(build_mode=True):
    audio_files = list(AUDIO_DIR.glob("*.mp3")) + list(AUDIO_DIR.glob("*.wav"))

    for audio_file in audio_files:
        json_path = OUTPUT_DIR / f"{audio_file.stem}.json"

        # Build 모드이면 이미 처리된 파일은 건너뛰기
        if build_mode and json_path.exists():
            print(f"⏭ 이미 처리됨 → {json_path.name}")
            continue

        print(f"🔊 처리 시작: {audio_file.name}")
        clean = denoise_audio(audio_file)
        final_segments = whisperx_diarization(clean)
        save_json(final_segments, audio_file)


# ======================
# 메인
# ======================
def main():
    process_all_audio(build_mode=True)


if __name__ == "__main__":
    main()
