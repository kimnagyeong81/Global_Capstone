import json
import shutil
from pathlib import Path

# ======================
# 프로젝트 루트 자동 인식
# ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data"
UPLOAD_DIR = DATA_DIR / "Uploads"
EMAIL_DIR = DATA_DIR / "Emails"
SLACK_DIR = DATA_DIR / "Slack"
VOICE_DIR = DATA_DIR / "voice" / "meeting voice datas"

VOICE_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

EMAIL_KEYS = {"from_name", "to_name", "subject", "body"}
SLACK_KEYS = {"channel_name", "channel_id", "user_name", "text"}


# ======================
# JSON 타입 판별
# ======================
def detect_json_type(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sample = data[0] if isinstance(data, list) and data else data
        if not isinstance(sample, dict):
            return "unknown"

        keys = set(sample.keys())

        if EMAIL_KEYS & keys:
            return "email"
        if SLACK_KEYS & keys:
            return "slack"

    except Exception as e:
        print(f"⚠ JSON 판별 실패: {path.name} ({e})")

    return "unknown"


# ======================
# 파일 이동
# ======================
def move(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.move(str(src), str(dst))
    print(f"✅ {src.name} → {dst.relative_to(PROJECT_ROOT)}")


# ======================
# 메인 처리
# ======================
def process_uploads():
    if not UPLOAD_DIR.exists():
        print(f"❌ Upload 디렉터리 없음: {UPLOAD_DIR}")
        return

    for file in UPLOAD_DIR.iterdir():
        if not file.is_file():
            continue

        ext = file.suffix.lower()

        # Voice
        if ext in VOICE_EXTS:
            move(file, VOICE_DIR)
            continue

        # JSON
        if ext == ".json":
            t = detect_json_type(file)
            if t == "email":
                move(file, EMAIL_DIR)
            elif t == "slack":
                move(file, SLACK_DIR)
            else:
                print(f"❓ 알 수 없는 JSON: {file.name}")
            continue

        print(f"⏭ 스킵됨: {file.name}")


if __name__ == "__main__":
    process_uploads()
