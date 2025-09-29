from pathlib import Path
from datetime import datetime
import json
from faster_whisper import WhisperModel

AUDIO_DIR = Path("data/audio")
OUT_PATH   = Path("data/voice_transcripts/transcripts.jsonl")

# 모델 크기: tiny / base / small / medium / large-v3
# CPU면 tiny/base 추천, 한국어 정확도는 small부터 꽤 좋아짐.
# 정확도 우선 → medium 권장 (CPU면 느릴 수 있음)
MODEL_NAME   = "medium"          # small → medium
COMPUTE_TYPE = "int8"            # CPU: int8 / GPU면 "int8_float16" 또는 "float16"

INITIAL_PROMPT = (
    "이 녹음은 한국어 회의입니다. 주제: 실험 설계, 피로(SMF), 설문, 납기, 일정, 예산, 담당자 배정, "
    "덕명동, 실험 참여자, 유튜브 쇼츠, 자막 속도, 정보 과부하."
)

def make_doc(audio_path: Path, text: str) -> dict:
    return {
        "doc_id": audio_path.stem,
        "source": "voice",
        "title": audio_path.stem,
        "created_at": datetime.fromtimestamp(audio_path.stat().st_mtime).isoformat(),
        "text": text
    }

def transcribe_file(model: WhisperModel, path: Path) -> str:
    segments, info = model.transcribe(
        str(path),
        language="ko",                 # 한국어 강제
        task="transcribe",
        beam_size=8,                   # 5~10에서 조절
        temperature=[0.0],
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt=INITIAL_PROMPT,
        condition_on_previous_text=True
    )
    return "".join(seg.text for seg in segments).strip()

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in sorted(AUDIO_DIR.glob("*.*")):
            print(f"[STT] {p.name} …")
            text = transcribe_file(model, p)
            if text:
                f.write(json.dumps(make_doc(p, text), ensure_ascii=False) + "\n")
    print(f"✅ saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
