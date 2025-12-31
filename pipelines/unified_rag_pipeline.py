# pipelines/unified_rag_pipeline.py  # 이 파일은 Streamlit 프론트엔드가 호출하는 “백엔드 RAG 파이프라인”입니다.
from __future__ import annotations  # 타입 힌트(자료형 표기)를 “문자열처럼” 지연 평가해서 순환 참조 문제를 줄여줍니다.

import os  # 운영체제(폴더/파일 경로 등) 기능을 쓰기 위한 표준 라이브러리입니다.
import json  # JSON 파일을 읽고/쓰는 표준 라이브러리입니다.
import glob  # 폴더에서 특정 패턴(예: *.json) 파일을 찾는 라이브러리입니다.
import time  # 시간(타임스탬프) 관련 기능을 사용합니다.
import math  # 수학 함수(올림/내림 등)를 사용합니다.
import hashlib  # 문자열을 해시(고유한 짧은 값)로 바꾸는 데 사용합니다.
from dataclasses import dataclass  # “문서(Doc)” 같은 데이터 구조를 편하게 만들기 위한 기능입니다.
from typing import Any, Dict, List, Optional, Tuple  # 타입 힌트(자료형 표기)에서 자주 쓰는 것들입니다.
from collections import defaultdict  # 딕셔너리를 편하게 초기화(기본값 자동 생성)하기 위해 사용합니다.

# -----------------------------  # 여기부터는 “외부 패키지”를 불러오는 구간입니다.
# chromadb / sentence_transformers / requests는 환경에 없을 수도 있으므로 try/except로 안전하게 처리합니다.
try:  # 아래 import가 실패해도 프로그램이 바로 죽지 않게 합니다.
    import chromadb  # 벡터DB(Chroma)를 사용하기 위한 패키지입니다.
    from chromadb.config import Settings as ChromaSettings  # Chroma 설정 객체를 가져옵니다.
except Exception:  # import 실패(설치 안 됨 등) 시
    chromadb = None  # chromadb가 없음을 표시(나중에 에러 메시지로 안내)
    ChromaSettings = None  # 설정 클래스도 없다고 표시합니다.

try:  # 문장 임베딩(텍스트를 벡터로 변환) 모델을 쓰기 위한 패키지 로드
    from sentence_transformers import SentenceTransformer  # 임베딩 모델 로더 클래스입니다.
except Exception:  # 설치가 안 되어있으면
    SentenceTransformer = None  # None으로 두고 나중에 안내합니다.

try:  # Ollama API를 HTTP로 호출하기 위한 requests 로드
    import requests  # HTTP 요청을 쉽게 보내기 위한 패키지입니다.
except Exception:  # 설치가 안 되어있으면
    requests = None  # None으로 두고 나중에 안내합니다.


# -----------------------------  # 설정값(상수)들을 모아둔 구역입니다.
# 기본 설정
# -----------------------------
DEFAULT_TOP_K = 10  # 검색에서 상위 몇 개 문서를 가져올지 기본값입니다(많을수록 근거가 늘지만, 너무 많으면 흐려질 수 있음).
MAX_CTX_CHARS = 12000  # LLM에게 넣을 근거 텍스트의 최대 길이(너무 길면 답이 산만해질 수 있어 제한).

DEFAULT_EMBED_MODEL = "BAAI/bge-m3"  # 기본 임베딩 모델 이름(텍스트를 숫자 벡터로 바꾸는 모델).
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Ollama 서버 주소(환경변수 없으면 기본값).
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")  # 사용할 LLM 모델명(환경변수 없으면 기본값).

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # 현재 파일 기준 “프로젝트 루트” 경로를 계산합니다.
DATA_ROOT = os.path.join(PROJECT_ROOT, "Data")  # 데이터가 들어있는 기본 폴더(Data) 경로를 지정합니다.

# 사용자 스크린샷 구조 반영: Data/Emails, Data/Slack, Data/Voice  # 실제 사용자 폴더명 대/소문자 차이를 흡수하기 위한 후보 목록입니다.
EMAIL_DIR_CANDIDATES = [  # 이메일 데이터가 들어있을 수 있는 폴더 후보 리스트입니다.
    os.path.join(DATA_ROOT, "Emails"),  # 흔한 폴더명 1
    os.path.join(DATA_ROOT, "Email"),   # 흔한 폴더명 2
    os.path.join(DATA_ROOT, "email"),   # 흔한 폴더명 3
]
SLACK_DIR_CANDIDATES = [  # 슬랙 데이터 폴더 후보 리스트입니다.
    os.path.join(DATA_ROOT, "Slack"),  # Slack
    os.path.join(DATA_ROOT, "slack"),  # slack
]
VOICE_DIR_CANDIDATES = [  # 음성 전사(Transcript) 데이터 폴더 후보 리스트입니다.
    # ✅ transcripts 하위가 아닐 수도 있으므로 Voice 루트도 후보에 포함  # 사용자 폴더 구조가 달라도 읽히도록 “여러 후보”를 둡니다.
    os.path.join(DATA_ROOT, "Voice"),               # Data/Voice
    os.path.join(DATA_ROOT, "voice"),               # Data/voice
    os.path.join(DATA_ROOT, "Voice", "transcripts"),# Data/Voice/transcripts
    os.path.join(DATA_ROOT, "voice", "transcripts"),# Data/voice/transcripts
]

# ✅ 요청사항: unified_BAAI_bge-m3을 기본 통합 벡터스토어 경로로 사용  # 통합 벡터DB 저장 폴더(Chroma persist directory)
DEFAULT_UNIFIED_DIR = os.path.join(PROJECT_ROOT, "vectorstores", "unified_BAAI_bge-m3")  # 통합 인덱스가 저장되는 폴더.

# 컬렉션 명은 기존 코드와 동일 유지(이미 만들어진 unified 인덱스와 호환)  # Chroma에서 “테이블 같은 단위”를 컬렉션이라 부릅니다.
PER_SOURCE_COLLECTION = "docs"  # 소스별(email/slack/voice) 인덱스가 저장될 컬렉션 이름.
UNIFIED_COLLECTION = "unified_docs"  # 통합 인덱스(모든 소스 합친 것)가 저장될 컬렉션 이름.


# -----------------------------  # 여기부터는 “공통 유틸 함수(도움 함수)” 모음입니다.
# 공통 유틸
# -----------------------------
def _ensure_deps():  # 필요한 외부 패키지가 설치되어 있는지 확인하는 함수입니다.
    if chromadb is None:  # chromadb를 import 못했다면(=설치 안 됐거나 문제)
        raise RuntimeError("chromadb가 설치되어 있지 않습니다. `pip install chromadb` 후 다시 시도해주세요.")  # 친절한 안내 메시지로 에러 발생.
    if SentenceTransformer is None:  # sentence-transformers가 없다면
        raise RuntimeError("sentence-transformers가 설치되어 있지 않습니다. `pip install sentence-transformers` 후 다시 시도해주세요.")  # 설치 안내.


def _resolve_embed_model(name: Optional[str]) -> str:  # 임베딩 모델 이름을 “정규화”해서 일관된 이름으로 바꿔줍니다.
    if not name:  # name이 None이거나 빈 문자열이면
        return DEFAULT_EMBED_MODEL  # 기본 모델을 반환합니다.

    s = str(name).strip()  # 입력값을 문자열로 만들고 앞뒤 공백 제거.
    low = s.lower()  # 소문자로 바꿔 비교를 쉽게 합니다.

    # 흔한 별칭/오입력 통일  # 사람들이 bge-m3를 여러 방식으로 적는 것을 하나로 통일합니다.
    if low in ("bge_m3", "bge-m3", "sentence-transformers/bge_m3", "sentence-transformers/bge-m3"):  # 흔한 표기들
        return "BAAI/bge-m3"  # 실제로 존재하는 공식 이름으로 통일.

    return s  # 위 케이스가 아니면 입력값을 그대로 사용합니다.


def _iter_json_files(root_dir: str) -> List[str]:  # 지정한 폴더 아래의 모든 json 파일 경로를 찾아 리스트로 반환합니다.
    if not root_dir or not os.path.isdir(root_dir):  # 폴더가 비어있거나 존재하지 않으면
        return []  # 빈 리스트 반환(=파일 없음).
    return sorted(glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True))  # 하위 폴더까지 재귀로 *.json 탐색 후 정렬.


def _safe_read_json(path: str) -> Any:  # JSON 파일을 “정상적으로” 읽는 함수입니다(실패 시 예외 발생).
    with open(path, "r", encoding="utf-8") as f:  # UTF-8로 파일을 열고
        return json.load(f)  # JSON 파싱 결과(딕셔너리/리스트 등)를 반환합니다.


def _read_json_safely(path: str) -> Any:  # JSON 파일 읽기를 “안전하게” 수행합니다(실패하면 None).
    try:  # 예외가 생길 수 있는 작업을 시도
        return _safe_read_json(path)  # 정상적으로 읽으면 그 결과를 반환
    except Exception:  # JSON이 깨졌거나 인코딩 문제 등 어떤 예외든
        return None  # None을 반환(=이 파일은 건너뜀).


def _as_text(x: Any) -> str:  # 어떤 값이든 “문자열 텍스트”로 바꿔주는 함수입니다.
    if x is None:  # 값이 None이면
        return ""  # 빈 문자열 반환.
    if isinstance(x, str):  # 이미 문자열이면
        return x  # 그대로 반환.
    return str(x)  # 그 외 타입은 str()로 문자열 변환.


def _pick_first(d: dict, keys: List[str]) -> Any:  # 딕셔너리에서 여러 키 후보 중 “처음 발견되는 값”을 반환합니다.
    for k in keys:  # 후보 키를 순서대로 검사
        if k in d and d[k] is not None:  # 키가 존재하고 값이 None이 아니면
            return d[k]  # 그 값을 반환.
    return None  # 아무 키도 못 찾으면 None 반환.


def _flatten_to_docs(obj: Any) -> List[dict]:
    """
    다양한 JSON 구조를 최대한 흡수해서 '문서 후보 dict 리스트'로 평탄화합니다.
    """
    if obj is None:  # JSON을 못 읽었거나 비어 있으면
        return []  # 문서 없음.

    if isinstance(obj, list):  # JSON 최상위가 리스트라면
        return [x for x in obj if isinstance(x, dict)]  # 그 중 딕셔너리(문서처럼 보이는 것)만 추립니다.

    if isinstance(obj, dict):  # JSON 최상위가 딕셔너리라면
        # 단일 문서 케이스  # 이 딕셔너리 자체가 문서(텍스트를 가진 1개 기록)일 수 있음
        if any(k in obj for k in ("text", "content", "body", "message", "transcript")):  # 이런 키가 하나라도 있으면
            return [obj]  # 이 딕셔너리를 “문서 1개”로 취급.

        # 흔한 컨테이너 키들  # 실제 데이터는 obj["messages"] 같은 리스트 안에 들어있는 경우가 많음
        container_keys = [  # 리스트를 담는 흔한 키들(경험적으로 많이 나오는 이름들)
            "messages", "items", "data", "logs", "events", "records", "threads",
            "emails", "conversations", "results"
        ]
        for ck in container_keys:  # 컨테이너 키 후보를 순회
            if ck in obj and isinstance(obj[ck], list):  # 해당 키가 있고, 그 값이 리스트면
                return [x for x in obj[ck] if isinstance(x, dict)]  # 리스트 내 딕셔너리들만 문서로 반환.

        # 딕셔너리 값 중 리스트가 있는 경우도 탐색  # 키 이름이 다양할 수 있으니 값 자체를 뒤져봄
        for v in obj.values():  # 딕셔너리의 모든 value를 확인
            if isinstance(v, list) and any(isinstance(x, dict) for x in v):  # value가 리스트이고 그 안에 dict가 있으면
                return [x for x in v if isinstance(x, dict)]  # dict 원소들만 반환.

    return []  # 어떤 패턴에도 해당하지 않으면 문서로 만들 수 없음.


def _first_existing_dir(paths: List[str]) -> Optional[str]:  # 후보 폴더들 중 “실제로 존재하는 첫 번째 폴더”를 반환합니다.
    for p in paths:  # 후보 폴더를 순서대로 확인
        if p and os.path.isdir(p):  # 경로가 있고, 실제 폴더로 존재하면
            return p  # 그 폴더를 반환.
    return None  # 아무 폴더도 없으면 None.


def _normalize_sources(sources: Optional[List[str]]) -> List[str]:  # 소스 이름을 email/slack/voice로 통일합니다.
    if not sources:  # sources가 None이거나 빈 리스트면
        return ["email", "slack", "voice"]  # 기본적으로 모든 소스를 사용.

    norm: List[str] = []  # 정규화된 결과를 담을 리스트
    for s in sources:  # 사용자가 선택한 소스들을 순회
        if not s:  # None/빈 값은 건너뜀
            continue
        s2 = s.strip().lower()  # 공백 제거 + 소문자 통일
        if s2 in ("emails", "email", "mail", "mails"):  # 이메일을 의미하는 다양한 표현
            norm.append("email")  # email로 통일
        elif s2 == "slack":  # slack은 그대로
            norm.append("slack")
        elif s2 in ("voice", "voices", "transcript", "transcripts"):  # 음성 전사 관련 표현
            norm.append("voice")  # voice로 통일
        else:
            norm.append(s2)  # 그 외는 일단 그대로 넣음(확장 가능)

    seen, out = set(), []  # 중복 제거를 위한 set(seen)과 결과 리스트(out)
    for s in norm:  # 정규화된 리스트를 다시 순회하면서
        if s not in seen:  # 아직 추가 안 된 값이면
            out.append(s)  # 결과에 추가
            seen.add(s)  # seen에도 기록
    return out  # 중복 제거된 소스 리스트 반환


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:  # 긴 텍스트를 여러 조각으로 나눕니다.
    text = (text or "").strip()  # None이면 ""로 바꾸고, 앞뒤 공백 제거
    if not text:  # 비어있으면
        return []  # chunk도 없음
    chunks = []  # 결과 chunk 리스트
    i = 0  # 현재 자를 위치(인덱스)
    while i < len(text):  # 텍스트 끝까지 반복
        chunks.append(text[i:i + chunk_size])  # chunk_size 만큼 잘라서 추가
        i += max(1, chunk_size - overlap)  # 다음 위치로 이동(겹침 overlap 만큼 덜 이동하여 문맥이 끊기는 문제를 완화)
    return chunks  # chunk 리스트 반환


def _sha1(s: str) -> str:  # 문자열을 SHA1 해시로 바꿔 “고유 ID” 비슷하게 만듭니다.
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()  # UTF-8 인코딩 후 해시를 16진수 문자열로 반환


def _manifest_path(persist_dir: str) -> str:  # persist_dir(벡터DB 폴더) 안의 manifest 파일 경로를 만들어줍니다.
    return os.path.join(persist_dir, "manifest.json")  # 예: vectorstores/unified_BAAI_bge-m3/manifest.json


def _load_manifest(persist_dir: str) -> Optional[Dict[str, Any]]:  # manifest.json을 읽어오는 함수입니다.
    mp = _manifest_path(persist_dir)  # manifest.json 경로 계산
    if not os.path.exists(mp):  # 파일이 없으면
        return None  # 없음 처리
    try:  # 파일 읽기 시도
        with open(mp, "r", encoding="utf-8") as f:  # UTF-8로 열고
            return json.load(f)  # JSON 파싱 후 반환
    except Exception:  # 파싱 실패/파일 깨짐 등
        return None  # None 반환


def _save_manifest(persist_dir: str, manifest: Dict[str, Any]) -> None:  # manifest.json을 저장하는 함수입니다.
    os.makedirs(persist_dir, exist_ok=True)  # 폴더가 없으면 생성(이미 있으면 무시)
    with open(_manifest_path(persist_dir), "w", encoding="utf-8") as f:  # manifest.json을 쓰기 모드로 열고
        json.dump(manifest, f, ensure_ascii=False, indent=2)  # 보기 좋게 들여쓰기해서 저장(한글도 깨지지 않게)


def _dataset_fingerprint(email_dir: Optional[str], slack_dir: Optional[str], voice_dir: Optional[str]) -> Dict[str, Any]:
    def file_list(root: Optional[str]) -> List[Dict[str, Any]]:  # 특정 폴더 아래 JSON 파일들의 목록/크기/수정시간을 모읍니다.
        if not root or not os.path.isdir(root):  # 폴더가 없으면
            return []  # 빈 리스트
        out = []  # 결과 리스트
        for fp in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):  # 하위 폴더까지 json 파일 모두 찾기
            try:
                st = os.stat(fp)  # 파일의 상태(크기, 수정시간 등)를 가져옴
                out.append({"path": fp, "size": int(st.st_size), "mtime": float(st.st_mtime)})  # 중요한 정보만 저장
            except Exception:
                continue  # 특정 파일에서 오류가 나면 그 파일은 건너뜀
        return out  # 파일 목록 반환

    return {  # email/slack/voice 각각에 대해 fingerprint 정보를 담아 반환
        "emails": file_list(email_dir),  # 이메일 폴더 지문
        "slack": file_list(slack_dir),   # 슬랙 폴더 지문
        "voice": file_list(voice_dir),   # 보이스 폴더 지문
    }


def _normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma metadata 제약:
    value는 str/int/float/bool/None만 가능.
    list/dict 등이 들어오면 upsert에서 ValueError가 납니다.
    """
    out: Dict[str, Any] = {}  # 정규화된 metadata를 담을 딕셔너리
    for k, v in (meta or {}).items():  # meta가 None이면 {}로 처리하고, key/value 순회
        if v is None:  # None이면 그대로 가능
            out[k] = None
        elif isinstance(v, (str, int, float, bool)):  # Chroma가 허용하는 타입이면
            out[k] = v  # 그대로 저장
        elif isinstance(v, list):  # 리스트는 Chroma가 저장 불가이므로
            out[k] = ", ".join(str(x) for x in v)  # 문자열로 합쳐서 저장(예: ["a","b"] -> "a, b")
        elif isinstance(v, dict):  # 딕셔너리도 저장 불가이므로
            out[k] = json.dumps(v, ensure_ascii=False)  # JSON 문자열로 변환해서 저장
        else:
            out[k] = str(v)  # 그 외 타입도 문자열로 변환해 저장
    return out  # 정규화된 metadata 반환


# -----------------------------  # 여기부터는 임베딩/Chroma 관련 기능입니다.
# 임베딩 / Chroma
# -----------------------------
_embedder_cache: Dict[str, Any] = {}  # 임베딩 모델을 캐싱(재사용)하기 위한 전역 딕셔너리입니다.


def _get_embedder(embed_model: Optional[str]) -> Any:  # 임베딩 모델을 가져오거나(없으면 생성) 반환합니다.
    _ensure_deps()  # 필요한 패키지 설치 여부 확인
    name = _resolve_embed_model(embed_model)  # 모델 이름을 정규화(별칭 통일)
    if name not in _embedder_cache:  # 캐시에 없으면
        _embedder_cache[name] = SentenceTransformer(name)  # 모델을 로드(시간이 걸릴 수 있음)
    return _embedder_cache[name]  # 캐시된 모델 반환


def _embed_texts(texts: List[str], embed_model: Optional[str]) -> List[List[float]]:  # 여러 텍스트를 임베딩 벡터로 변환합니다.
    model = _get_embedder(embed_model)  # 임베딩 모델 가져오기
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)  # 텍스트 -> 벡터 (정규화하여 거리 계산이 안정적)
    return [v.tolist() for v in vecs]  # numpy 배열을 파이썬 리스트로 변환(저장/전송에 편함)


def _get_chroma_client(persist_dir: str):  # Chroma(벡터DB) 클라이언트를 생성합니다.
    _ensure_deps()  # 패키지 설치 확인
    os.makedirs(persist_dir, exist_ok=True)  # 벡터DB 폴더가 없으면 생성
    return chromadb.PersistentClient(  # “디스크에 저장되는” Chroma 클라이언트 생성
        path=persist_dir,  # 이 경로 아래에 DB 파일이 저장됩니다.
        settings=ChromaSettings(anonymized_telemetry=False),  # 텔레메트리(사용 통계) 비활성화
    )


def _get_or_create_collection(client, name: str):  # 특정 컬렉션을 가져오거나, 없으면 새로 생성합니다.
    try:
        return client.get_collection(name=name)  # 컬렉션이 이미 있으면 가져오기
    except Exception:
        return client.create_collection(name=name, metadata={"created_at": time.time()})  # 없으면 생성(생성 시각 기록)


def _collection_has_data(persist_dir: str, collection_name: str) -> bool:
    """
    ✅ 기존 _looks_indexed는 '폴더에 파일이 있으면 True'라서
       실제 컬렉션이 비어 있어도 인덱싱된 것으로 오판할 수 있습니다.
    -> 컬렉션에 ids가 실제로 존재하는지로 판단합니다.
    """
    try:
        client = _get_chroma_client(persist_dir)  # 해당 persist_dir의 Chroma를 엽니다.
        col = _get_or_create_collection(client, collection_name)  # 컬렉션을 가져오거나 만듭니다.
        got = col.get(include=[])  # 컬렉션에서 id 목록을 가져옵니다(include=[]면 데이터 본문은 안 가져와서 가볍습니다)
        return bool(got and got.get("ids"))  # ids가 하나라도 있으면 True(데이터 존재)
    except Exception:
        return False  # 오류가 나면 “데이터 없음”으로 처리


# -----------------------------  # 여기부터는 실제 JSON 데이터를 Doc 객체로 로드하는 부분입니다.
# 데이터 로딩
# -----------------------------
@dataclass
class Doc:  # “하나의 문서(기록)”를 표현하는 자료구조입니다.
    doc_id: str  # 문서 고유 ID(중복 방지용)
    text: str  # 문서 본문 텍스트(임베딩 대상)
    title: str  # 문서 제목(메일 제목/슬랙 채널 등)
    source_type: str  # email | slack | voice  # 어떤 소스에서 왔는지 표시
    metadata: Dict[str, Any]  # 추가 정보(파일 경로, 날짜, 보낸 사람 등)


def _load_email_docs() -> List[Doc]:  # 이메일 JSON들을 읽어서 Doc 리스트로 만드는 함수입니다.
    email_dir = _first_existing_dir(EMAIL_DIR_CANDIDATES)  # 후보 폴더 중 실제 존재하는 폴더를 선택
    if not email_dir:  # 폴더가 없으면
        return []  # 이메일 문서 없음

    docs: List[Doc] = []  # 결과 Doc 리스트
    for fp in _iter_json_files(email_dir):  # 이메일 폴더 아래 모든 json 파일 경로 순회
        obj = _read_json_safely(fp)  # json 읽기(실패하면 None)
        if obj is None:  # 읽기 실패면
            continue  # 이 파일은 건너뜀

        items = _flatten_to_docs(obj)  # json 구조를 “문서 dict 리스트”로 평탄화
        for idx, item in enumerate(items):  # 각 문서 item 순회
            subject = _as_text(_pick_first(item, ["subject", "title", "topic"])) or os.path.basename(fp)  # 제목 후보를 찾아 제목 결정
            body = _as_text(_pick_first(item, ["body", "content", "text", "message"])).strip()  # 본문 후보를 찾아 본문 결정

            # emails.json 내부에 messages 배열로 들어온 경우도 흡수  # 어떤 JSON은 본문이 messages 배열에 들어있을 수 있음
            if not body and isinstance(item.get("messages"), list):  # body가 비어있고 messages가 리스트면
                body = "\n\n".join(  # messages들의 텍스트를 이어붙여 body를 만듭니다.
                    _as_text(_pick_first(m, ["text", "body", "content", "message"]))  # 메시지에서 텍스트 후보 키들을 탐색
                    for m in item["messages"]  # messages 순회
                    if isinstance(m, dict)  # dict인 것만 처리
                ).strip()  # 앞뒤 공백 제거

            if not body:  # 본문이 여전히 비어있으면(텍스트 없음)
                continue  # 문서로 만들 의미가 없으니 건너뜀

            meta = dict(item.get("metadata") or {})  # item["metadata"]가 있으면 가져오고, 없으면 빈 dict
            meta.setdefault("path", fp)  # 파일 경로 기록(없으면 설정)
            meta.setdefault("filename", os.path.basename(fp))  # 파일명 기록(없으면 설정)
            meta.setdefault("source_type", "email")  # 소스 타입 기록
            for k in ["date", "timestamp", "from", "to", "cc", "department", "thread_id", "id", "message_id"]:  # 자주 쓰는 추가 필드들
                if k in item and k not in meta:  # item에 있는데 meta에는 없으면
                    meta[k] = item.get(k)  # meta로 복사

            base = _as_text(_pick_first(item, ["id", "message_id", "thread_id"])) or f"{_sha1(fp)}_{idx}"  # 고유값 후보(없으면 파일해시+인덱스)
            doc_id = f"email::{_sha1(base)[:16]}::{_sha1(fp)[:8]}::{idx}"  # doc_id 생성(충돌 방지 위해 여러 조각 조합)

            docs.append(Doc(doc_id=doc_id, text=body, title=subject, source_type="email", metadata=meta))  # Doc 객체로 추가
    return docs  # 이메일 문서 리스트 반환


def _load_slack_docs() -> List[Doc]:  # 슬랙 JSON들을 읽어서 Doc 리스트로 만드는 함수입니다.
    slack_dir = _first_existing_dir(SLACK_DIR_CANDIDATES)  # 슬랙 폴더 선택
    if not slack_dir:  # 폴더가 없으면
        return []  # 슬랙 문서 없음

    docs: List[Doc] = []  # 결과 Doc 리스트
    for fp in _iter_json_files(slack_dir):  # 슬랙 폴더의 모든 json 파일 순회
        obj = _read_json_safely(fp)  # json 읽기
        if obj is None:  # 실패하면
            continue  # 건너뜀

        items = _flatten_to_docs(obj)  # 구조 평탄화
        for idx, item in enumerate(items):  # 각 메시지(문서) 순회
            channel = _as_text(_pick_first(item, ["channel", "room", "channel_name"])) or "slack"  # 채널 이름 후보 탐색
            user = _as_text(_pick_first(item, ["user", "author", "username"])) or ""  # 작성자 후보 탐색
            text = _as_text(_pick_first(item, ["text", "content", "message", "body"])).strip()  # 메시지 텍스트 후보 탐색
            if not text:  # 텍스트가 없으면
                continue  # 건너뜀

            meta = dict(item.get("metadata") or {})  # meta가 있으면 가져오고 없으면 빈 dict
            meta.setdefault("path", fp)  # 파일 경로 저장
            meta.setdefault("filename", os.path.basename(fp))  # 파일명 저장
            meta.setdefault("source_type", "slack")  # 소스 타입 저장
            for k in ["ts", "timestamp", "date", "thread_ts", "department"]:  # 슬랙에서 흔한 시간/스레드 관련 키
                if k in item and k not in meta:  # item에 있고 meta에는 없으면
                    meta[k] = item.get(k)  # meta로 복사
            if user and "user" not in meta:  # user 정보가 있고 meta에 user가 없으면
                meta["user"] = user  # 작성자 저장
            if channel and "channel" not in meta:  # channel 정보가 있고 meta에 channel이 없으면
                meta["channel"] = channel  # 채널 저장

            base = _as_text(_pick_first(item, ["id", "ts"])) or f"{_sha1(fp)}_{idx}"  # 고유값 후보(없으면 파일해시+인덱스)
            doc_id = f"slack::{_sha1(base)[:16]}::{_sha1(fp)[:8]}::{idx}"  # 슬랙 문서 ID 생성
            title = f"#{channel}"  # title은 보통 채널명으로 설정(#general 같은 형태)

            docs.append(Doc(doc_id=doc_id, text=text, title=title, source_type="slack", metadata=meta))  # Doc 추가
    return docs  # 슬랙 문서 리스트 반환


def _load_voice_docs() -> List[Doc]:  # 음성 전사(Transcript) JSON들을 읽어 Doc 리스트로 만듭니다.
    voice_dir = _first_existing_dir(VOICE_DIR_CANDIDATES)  # Voice 폴더 후보 중 실제 존재 폴더 선택
    if not voice_dir:  # 폴더가 없으면
        return []  # voice 문서 없음

    docs: List[Doc] = []  # 결과 Doc 리스트
    for fp in _iter_json_files(voice_dir):  # voice 폴더의 모든 json 파일 순회
        obj = _read_json_safely(fp)  # json 읽기
        if obj is None:  # 실패하면
            continue  # 건너뜀

        items = _flatten_to_docs(obj)  # 평탄화 시도
        if not items and isinstance(obj, dict):  # 평탄화 결과가 없는데 최상위가 dict면
            items = [obj]  # 그 dict 자체를 문서 1개로 취급

        for idx, item in enumerate(items):  # 각 음성 기록 순회
            text = _as_text(_pick_first(item, ["text", "content", "transcript", "body", "message"])).strip()  # 전사 텍스트 후보 키
            if not text:  # 텍스트가 없으면
                continue  # 건너뜀

            meta = dict(item)  # voice는 item 자체에 유용한 정보가 많아서 통째로 meta로 시작
            meta.setdefault("path", fp)  # 파일 경로 저장
            meta.setdefault("filename", os.path.basename(fp))  # 파일명 저장
            meta.setdefault("source_type", "voice")  # 소스 타입 저장

            title = f"voice:{os.path.basename(fp)}"  # 제목은 파일명을 기반으로 표시
            base = _as_text(_pick_first(item, ["id", "session_id"])) or f"{_sha1(fp)}_{idx}"  # 고유값 후보
            doc_id = f"voice::{_sha1(base)[:16]}::{_sha1(fp)[:8]}::{idx}"  # voice 문서 ID 생성

            docs.append(Doc(doc_id=doc_id, text=text, title=title, source_type="voice", metadata=meta))  # Doc 추가
    return docs  # voice 문서 리스트 반환


def _load_docs_by_source(source_type: str) -> List[Doc]:  # 소스 타입 문자열에 맞는 로더 함수를 호출하는 “분기 함수”입니다.
    if source_type == "email":  # email이면
        return _load_email_docs()  # 이메일 로더 실행
    if source_type == "slack":  # slack이면
        return _load_slack_docs()  # 슬랙 로더 실행
    if source_type == "voice":  # voice이면
        return _load_voice_docs()  # 보이스 로더 실행
    return []  # 그 외 소스는 지원하지 않으므로 빈 리스트


# -----------------------------  # 여기부터는 “인덱싱(벡터DB에 저장)” 관련입니다.
# 인덱싱
# -----------------------------
def _upsert_docs_into_collection(collection, docs: List[Doc], embed_model: Optional[str]) -> int:  # 문서를 chunk로 나누고 임베딩 후 DB에 upsert합니다.
    if not docs:  # 문서가 없으면
        return 0  # 저장할 chunk도 0

    ids: List[str] = []  # Chroma에 넣을 각 chunk의 고유 ID 리스트
    texts: List[str] = []  # 각 chunk의 텍스트 리스트
    metas: List[Dict[str, Any]] = []  # 각 chunk의 메타데이터 리스트

    n_chunks = 0  # 총 몇 개 chunk를 만들었는지 세는 카운터
    for d in docs:  # 문서별로 반복
        chunks = _chunk_text(d.text)  # 문서 본문을 여러 chunk로 자르기
        for ci, ch in enumerate(chunks):  # chunk index(ci)와 chunk text(ch) 반복
            ids.append(f"{d.doc_id}::chunk{ci}")  # chunk별 ID 생성(문서ID + chunk번호)
            texts.append(ch)  # chunk 텍스트 저장
            meta = dict(d.metadata)  # 문서 메타데이터 복사(원본 보호)
            meta.update({"source_type": d.source_type, "title": d.title, "doc_id": d.doc_id, "chunk_index": ci})  # chunk 관련 필드 추가
            metas.append(_normalize_metadata(meta))  # Chroma 제약에 맞게 metadata 정규화
            n_chunks += 1  # chunk 개수 증가

    vecs = _embed_texts(texts, embed_model)  # 모든 chunk 텍스트를 임베딩(벡터로 변환)
    collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=vecs)  # Chroma에 저장/갱신(upsert)
    return n_chunks  # 저장한 chunk 개수 반환


def build_indexes(  # 실제로 벡터DB(소스별 + 통합)를 “새로 구성”하는 함수입니다.
    embed_model: Optional[str] = None,  # 사용할 임베딩 모델(없으면 기본)
    vectorstore_paths: Optional[Dict[str, str]] = None,  # 소스별 벡터DB 경로를 사용자 정의로 주고 싶을 때 사용
    unified_vectorstore_path: Optional[str] = None,  # 통합 벡터DB 경로(없으면 기본 DEFAULT_UNIFIED_DIR)
    sources: Optional[List[str]] = None,  # 사용할 소스들(체크박스 선택 결과)
    collection_name: str = PER_SOURCE_COLLECTION,  # 소스별 컬렉션명
    unified_collection_name: str = UNIFIED_COLLECTION,  # 통합 컬렉션명
) -> Dict[str, Any]:
    _ensure_deps()  # 패키지 설치 확인
    embed_model = _resolve_embed_model(embed_model)  # 임베딩 모델 이름 통일
    norm_sources = _normalize_sources(sources)  # 소스 이름 통일 + 중복 제거

    vectorstore_paths = vectorstore_paths or {}  # None이면 빈 dict로 처리

    stats = {"per_source": {}, "unified": {}}  # 처리 결과(문서/청크 수 등)를 담을 통계 딕셔너리

    all_docs: List[Doc] = []  # 통합 인덱스를 만들기 위해 모든 문서를 합쳐 담는 리스트
    for src in norm_sources:  # 선택된 소스들(email/slack/voice)을 순회
        docs = _load_docs_by_source(src)  # 해당 소스 문서 로딩
        all_docs.extend(docs)  # 통합 리스트에도 추가

        persist_dir = vectorstore_paths.get(src) or os.path.join(PROJECT_ROOT, "vectorstores", f"{src}_BAAI_bge-m3")  # 소스별 저장 폴더 결정
        client = _get_chroma_client(persist_dir)  # 해당 폴더의 Chroma 열기
        col = _get_or_create_collection(client, collection_name)  # 컬렉션 가져오기/생성

        # 기존 데이터 제거 후 재삽입  # “완전 재생성”을 위해 기존 ids를 지우고 다시 넣습니다.
        try:
            existing = col.get(include=[])  # ids만 가져옴(가볍게)
            if existing and existing.get("ids"):  # 기존 ids가 있다면
                col.delete(ids=existing["ids"])  # 모두 삭제
        except Exception:
            pass  # 삭제 중 오류가 나도 일단 진행(최악의 경우 기존 데이터가 남아있을 수 있음)

        inserted = _upsert_docs_into_collection(col, docs, embed_model)  # 문서들을 임베딩/저장
        stats["per_source"][src] = {"docs": len(docs), "chunks": inserted, "persist_dir": persist_dir}  # 통계 기록

    # ✅ 통합 인덱스는 unified_BAAI_bge-m3 기본  # 통합 저장 폴더를 결정합니다.
    uni_dir = unified_vectorstore_path or DEFAULT_UNIFIED_DIR  # 인자가 없으면 기본 경로 사용
    uni_client = _get_chroma_client(uni_dir)  # 통합 Chroma 열기
    uni_col = _get_or_create_collection(uni_client, unified_collection_name)  # 통합 컬렉션 열기/생성

    try:
        existing = uni_col.get(include=[])  # 통합 컬렉션의 ids 확인
        if existing and existing.get("ids"):  # 기존 데이터가 있으면
            uni_col.delete(ids=existing["ids"])  # 삭제 후
    except Exception:
        pass  # 오류 시 무시(하지만 가능하면 정상적으로 삭제되도록 환경을 맞추는 것이 좋습니다)

    inserted = _upsert_docs_into_collection(uni_col, all_docs, embed_model)  # 통합 문서들을 임베딩/저장
    stats["unified"] = {"docs": len(all_docs), "chunks": inserted, "persist_dir": uni_dir}  # 통합 통계 기록
    return stats  # 전체 통계 반환


def ensure_indexes(  # “데이터가 바뀌면 자동으로 재인덱싱”을 수행하는 함수입니다.
    *,
    embed_model: Optional[str] = None,  # 임베딩 모델
    vectorstore_paths: Optional[Dict[str, str]] = None,  # 소스별 벡터스토어 경로들
    unified_vectorstore_path: Optional[str] = None,  # 통합 벡터스토어 경로
    sources: Optional[List[str]] = None,  # 선택 소스(체크박스 결과)
    force_rebuild: bool = False,  # True면 무조건 다시 만듭니다.
) -> Dict[str, Any]:
    """
    streamlit_app.py 시작 시 호출용:
    - 데이터 fingerprint 비교
    - 바뀌면 자동 build_indexes()
    """
    _ensure_deps()  # 패키지 설치 확인
    embed_model = _resolve_embed_model(embed_model)  # 임베딩 모델 이름 통일
    norm_sources = _normalize_sources(sources)  # 소스 이름 통일

    # ✅ 통합 인덱스 경로 기본값은 unified_BAAI_bge-m3  # 통합 벡터스토어 경로 결정
    unified_vectorstore_path = unified_vectorstore_path or DEFAULT_UNIFIED_DIR  # None이면 기본 경로

    email_dir = _first_existing_dir(EMAIL_DIR_CANDIDATES)  # 이메일 폴더 실제 경로
    slack_dir = _first_existing_dir(SLACK_DIR_CANDIDATES)  # 슬랙 폴더 실제 경로
    voice_dir = _first_existing_dir(VOICE_DIR_CANDIDATES)  # 보이스 폴더 실제 경로

    fp = _dataset_fingerprint(email_dir, slack_dir, voice_dir)  # 현재 데이터 상태(파일 목록/크기/mtime)로 fingerprint 생성

    new_manifest = {  # 이번 실행 시점의 상태를 manifest로 구성
        "embed_model": embed_model,  # 어떤 임베딩 모델을 사용했는지
        "sources": norm_sources,  # 어떤 소스를 사용했는지
        "data_fingerprint": fp,  # 데이터 파일들의 상태
        "generated_at": time.time(),  # 이 manifest가 만들어진 시간
        "email_dir": email_dir,  # 이메일 폴더 경로
        "slack_dir": slack_dir,  # 슬랙 폴더 경로
        "voice_dir": voice_dir,  # 보이스 폴더 경로
    }

    old = _load_manifest(unified_vectorstore_path)  # 기존 manifest(지난번 실행 때 저장된 상태)를 읽어옵니다.

    def same(a: Dict[str, Any], b: Dict[str, Any]) -> bool:  # 두 manifest가 동일한지 비교하는 내부 함수
        if not a or not b:  # 둘 중 하나라도 비어있으면
            return False  # 동일하다고 볼 수 없음
        return (a.get("embed_model") == b.get("embed_model")  # 임베딩 모델이 같고
                and a.get("sources") == b.get("sources")  # 소스 선택이 같고
                and a.get("data_fingerprint") == b.get("data_fingerprint"))  # 데이터 fingerprint가 같으면 “동일”

    if force_rebuild:  # 강제 재빌드 옵션이 켜져 있으면
        reason = "force_rebuild=True"  # 재생성 이유 기록
    elif old is None:  # 기존 manifest가 없으면(처음 실행 등)
        reason = "manifest_missing"  # 재생성해야 함
    elif not same(old, new_manifest):  # 데이터/설정이 바뀌었으면
        reason = "data_changed"  # 재생성해야 함
    else:
        # ✅ “폴더에 파일이 있다”가 아니라, 실제 unified 컬렉션에 데이터가 있는지도 확인  # DB 폴더는 있는데 내용이 비었을 수 있음
        if not _collection_has_data(unified_vectorstore_path, UNIFIED_COLLECTION):  # 컬렉션에 실제 ids가 없으면
            reason = "collection_empty"  # 재생성해야 함
        else:
            return {"did_rebuild": False, "reason": "up_to_date", "stats": {}}  # 여기까지 왔으면 최신 상태이므로 재생성 안 함

    stats = build_indexes(  # 재생성이 필요하면 build_indexes 실행
        embed_model=embed_model,  # 임베딩 모델
        vectorstore_paths=vectorstore_paths,  # 소스별 경로(있다면)
        unified_vectorstore_path=unified_vectorstore_path,  # 통합 경로
        sources=norm_sources,  # 선택 소스
    )
    _save_manifest(unified_vectorstore_path, new_manifest)  # 새 manifest를 저장(다음 실행 때 비교용)
    return {"did_rebuild": True, "reason": reason, "stats": stats}  # 재생성 결과 반환


def debug_data_counts() -> dict:  # 데이터가 실제로 로드되는지 확인하기 위한 디버깅 함수입니다.
    email_dir = _first_existing_dir(EMAIL_DIR_CANDIDATES)  # 이메일 폴더 찾기
    slack_dir = _first_existing_dir(SLACK_DIR_CANDIDATES)  # 슬랙 폴더 찾기
    voice_dir = _first_existing_dir(VOICE_DIR_CANDIDATES)  # 보이스 폴더 찾기

    email_files = _iter_json_files(email_dir) if email_dir else []  # 이메일 json 파일 개수
    slack_files = _iter_json_files(slack_dir) if slack_dir else []  # 슬랙 json 파일 개수
    voice_files = _iter_json_files(voice_dir) if voice_dir else []  # 보이스 json 파일 개수

    email_docs = _load_email_docs()  # 이메일 문서 로딩 결과
    slack_docs = _load_slack_docs()  # 슬랙 문서 로딩 결과
    voice_docs = _load_voice_docs()  # 보이스 문서 로딩 결과

    return {  # Streamlit UI나 로그에 찍어서 확인하기 좋은 형태로 반환
        "dirs": {"email": email_dir, "slack": slack_dir, "voice": voice_dir},  # 실제로 선택된 폴더 경로
        "json_files": {"email": len(email_files), "slack": len(slack_files), "voice": len(voice_files)},  # 파일 개수
        "docs_loaded": {"email": len(email_docs), "slack": len(slack_docs), "voice": len(voice_docs)},  # 로딩된 문서 개수
        "samples": {  # 문서가 로딩되면 “첫 문서 제목”을 샘플로 보여줌(정상 로딩 여부 빠른 확인)
            "email": (email_docs[0].title[:80] if email_docs else None),  # 이메일 첫 제목
            "slack": (slack_docs[0].title[:80] if slack_docs else None),  # 슬랙 첫 제목
            "voice": (voice_docs[0].title[:80] if voice_docs else None),  # 보이스 첫 제목
        }
    }


# -----------------------------  # 여기부터는 “검색(벡터DB에서 관련 chunk 찾기)” 파트입니다.
# 검색
# -----------------------------
def _dist_to_score(dist: float) -> float:  # Chroma의 distance 값을 사람이 보기 쉬운 score로 변환합니다.
    try:
        d = float(dist)  # 숫자로 변환 시도
    except Exception:
        return 0.0  # 변환 실패하면 점수 0
    return float(1.0 / (1.0 + max(0.0, d)))  # distance가 작을수록 score가 커지도록 변환(0~1 사이로 안정)


def _query_collection(  # 특정 컬렉션에서 질문과 가까운 문서를 top_k개 찾습니다.
    persist_dir: str,  # 벡터DB가 저장된 폴더
    collection_name: str,  # 검색할 컬렉션명
    query_text: str,  # 사용자 질문 텍스트
    top_k: int,  # 몇 개 결과를 가져올지
    embed_model: Optional[str],  # 사용할 임베딩 모델
    where: Optional[Dict[str, Any]] = None,  # 메타데이터 필터(예: source_type='email')
) -> List[Dict[str, Any]]:
    client = _get_chroma_client(persist_dir)  # 해당 폴더의 Chroma 열기
    col = _get_or_create_collection(client, collection_name)  # 컬렉션 열기/생성

    qv = _embed_texts([query_text], embed_model)[0]  # 질문을 임베딩 벡터로 변환(리스트로 넣었으니 [0]으로 첫 벡터 선택)
    res = col.query(  # Chroma에게 “이 벡터와 가까운 문서”를 찾아달라고 요청
        query_embeddings=[qv],  # 쿼리 벡터(리스트 형태로 전달)
        n_results=top_k,  # 결과 개수
        where=where,  # 필터 조건(없으면 전체)
        include=["documents", "metadatas", "distances"],  # 결과로 문서 텍스트/메타/거리 가져오기
    )

    docs = (res.get("documents") or [[]])[0]  # 결과 문서 텍스트(첫 쿼리에 대한 결과)
    metas = (res.get("metadatas") or [[]])[0]  # 결과 메타데이터
    dists = (res.get("distances") or [[]])[0]  # 결과 거리(distance)

    out: List[Dict[str, Any]] = []  # 최종 결과를 담을 리스트
    for doc, meta, dist in zip(docs, metas, dists):  # docs/metas/dists를 묶어서 순회
        meta = meta or {}  # meta가 None이면 빈 dict
        full = doc or ""  # doc이 None이면 빈 문자열
        out.append({  # Streamlit/UI에서 쓰기 좋은 형태로 결과를 정리
            "source": str(meta.get("source_type") or "unknown"),  # 어떤 소스인지(email/slack/voice)
            "title": str(meta.get("title") or "context"),  # 제목(메일 제목/채널 등)
            "text": full,                      # ✅ 프롬프트에는 full text를 사용  # LLM에 넣을 때는 전체 텍스트가 좋음
            "snippet": full[:1000],            # UI용 snippet은 짧게  # 화면에는 너무 길면 불편하니 1000자만 보여줌
            "metadata": meta,  # 원본 메타데이터
            "score": _dist_to_score(dist),  # distance를 score로 변환
        })
    return out  # 검색 결과 리스트 반환


def _query_unified_with_source_filter(  # 통합 컬렉션에서 “선택된 소스만” 검색하도록 처리합니다.
    persist_dir: str,  # 통합 벡터DB 폴더
    query_text: str,  # 질문
    top_k: int,  # 결과 개수
    embed_model: Optional[str],  # 임베딩 모델
    norm_sources: List[str],  # 선택 소스 목록(정규화된 email/slack/voice)
) -> List[Dict[str, Any]]:
    # where $in 지원 버전이면 사용, 아니면 source별로 쪼개서 merge  # Chroma 버전에 따라 $in이 안 될 수 있으므로 대비합니다.
    try:
        where = {"source_type": {"$in": norm_sources}}  # source_type이 선택된 소스 중 하나인 것만 검색
        return _query_collection(persist_dir, UNIFIED_COLLECTION, query_text, top_k, embed_model, where=where)  # 통합 컬렉션에서 필터 검색
    except Exception:
        merged: List[Dict[str, Any]] = []  # $in이 실패하면 source별 결과를 합칠 리스트
        each_k = max(1, math.ceil(top_k / max(1, len(norm_sources))))  # 소스별로 몇 개씩 가져올지 계산
        for s in norm_sources:  # 각 소스별로
            merged.extend(_query_collection(persist_dir, UNIFIED_COLLECTION, query_text, each_k, embed_model, where={"source_type": s}))  # 각각 검색해서 합치기
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)  # 점수 높은 순으로 정렬
        return merged[:top_k]  # top_k개만 잘라 반환


# -----------------------------  # 여기부터는 “답변 생성(LLM 호출)” 파트입니다.
# 답변 생성(품질 개선 핵심)
# -----------------------------
def _build_prompt_high_quality(question: str, contexts: List[Dict[str, Any]]) -> str:
    """
    - Group contexts by source
    - Truncate to MAX_CTX_CHARS to avoid overload
    - Encourage evidence-grounded, polite, easy-to-read English answers (not a report)
    """
    grouped: Dict[str, List[str]] = defaultdict(list)

    for c in contexts:
        src = c.get("source", "unknown")
        title = c.get("title", "context")
        text = (c.get("text") or "").strip()
        if not text:
            continue
        grouped[src].append(f'""{title}""\n{text}')

    ordered_sources = ["slack", "voice", "email"]
    lines: List[str] = []
    for src in ordered_sources:
        if src not in grouped:
            continue
        lines.append(f"[{src.upper()}]")
        lines.append("\n\n".join(grouped[src]))

    context_blob = "\n\n".join(lines).strip()
    context_blob = context_blob[:MAX_CTX_CHARS]

    return (
        # ✅ 핵심 요구사항 1: 영어로 답변 고정
        "Please answer in English only.\n"
        "\n"
        # ✅ 핵심 요구사항 2: 보고서 말투 금지 + 존댓말 느낌(정중한 톤)
        "You are a helpful assistant. Please respond politely and naturally, like you are speaking to a user.\n"
        "Do NOT write in a formal report style. Avoid headings like 'Summary/Analysis/Conclusion' and avoid rigid numbering.\n"
        "\n"
        # ✅ 근거 기반 규칙(환각 방지)
        "Use ONLY the information in the provided sources. If something is not clearly supported, say so.\n"
        "If you need to make an assumption, clearly label it as an assumption.\n"
        "\n"
        # ✅ 읽기 쉬운 출력 가이드(강제 번호 대신 부드러운 구조)
        "Please keep the answer easy to read:\n"
        "- Start with a short direct answer (1–3 sentences).\n"
        "- Then explain briefly using evidence from the sources.\n"
        "- When you cite evidence, mention which source it came from (Slack / Voice / Email).\n"
        "- End with a helpful next step or suggestion if appropriate.\n"
        "\n"
        f"Question:\n{question}\n\n"
        f"Sources:\n{context_blob}\n\n"
        "Answer (English only):"
    )


def _call_ollama(prompt: str, model: str) -> str:  # Ollama 로컬 LLM에 HTTP 요청을 보내 답변을 받는 함수입니다.
    if requests is None:  # requests가 설치 안 되어 있으면
        raise RuntimeError("requests가 설치되어 있지 않아 Ollama 호출을 할 수 없습니다. `pip install requests`")  # 설치 안내
    url = f"{DEFAULT_OLLAMA_BASE_URL}/api/generate"  # Ollama generate API 주소
    r = requests.post(  # HTTP POST 요청 전송
        url,  # 요청 주소
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},  # 요청 본문(모델/프롬프트/옵션)
        timeout=180  # 180초 안에 응답이 없으면 타임아웃
    )
    r.raise_for_status()  # HTTP 오류 코드면 예외 발생(문제 원인을 빨리 알 수 있음)
    j = r.json()  # 응답 JSON 파싱
    return str(j.get("response") or "").strip()  # response 필드를 꺼내서 문자열로 반환(없으면 빈 문자열)


def _generate_answer(question: str, contexts: List[Dict[str, Any]], llm_model: Optional[str]) -> str:  # 질문+근거로 최종 답변을 생성합니다.
    prompt = _build_prompt_high_quality(question, contexts)  # 좋은 프롬프트 생성
    return _call_ollama(prompt, llm_model or DEFAULT_OLLAMA_MODEL)  # Ollama 호출(모델이 없으면 기본 모델)


# -----------------------------  # 여기부터는 “프론트엔드가 실제로 호출하는 공개 함수”들입니다.
# 메인 Query API (streamlit 호환)
# -----------------------------
def query(  # Streamlit이 호출하는 메인 함수: 질문을 받으면 답변과 근거 리스트를 반환합니다.
    question: str,  # 사용자 질문
    sources: Optional[List[str]] = None,  # 사f용할 데이터 소스 목록(체크박스 결과)
    top_k: int = DEFAULT_TOP_K,  # 검색 결과 개수
    embed_model: Optional[str] = None,  # 임베딩 모델
    llm_model: Optional[str] = None,  # LLM 모델(Ollama)
    vectorstore_paths: Optional[Dict[str, str]] = None,  # 소스별 벡터스토어 경로(고급 옵션)
    unified_vectorstore_path: Optional[str] = None,  # 통합 벡터스토어 경로
) -> Tuple[str, List[Dict[str, Any]]]:
    _ensure_deps()  # 패키지 설치 확인

    question = (question or "").strip()  # 질문이 None이면 ""로 만들고 공백 제거
    if not question:  # 질문이 비어있으면
        return "질문이 비어 있습니다.", []  # 안내 메시지와 빈 근거 반환

    norm_sources = _normalize_sources(sources)  # 소스 선택을 email/slack/voice로 통일
    top_k = int(top_k) if top_k else DEFAULT_TOP_K  # top_k가 이상하게 들어오면 기본값 사용
    embed_model = _resolve_embed_model(embed_model)  # 임베딩 모델 이름 통일

    # ✅ unified 기본 경로: unified_BAAI_bge-m3  # 통합 벡터스토어 경로 결정
    unified_vectorstore_path = unified_vectorstore_path or DEFAULT_UNIFIED_DIR  # None이면 기본 경로 사용

    # ✅ 인덱스가 없거나, 컬렉션이 비어 있으면 자동 재생성  # 사용자가 수동으로 인덱싱하지 않아도 되도록 자동화
    if not _collection_has_data(unified_vectorstore_path, UNIFIED_COLLECTION):  # 통합 컬렉션에 데이터가 없으면
        ensure_indexes(  # 데이터 상태를 확인하고 필요하면 인덱스를 빌드
            embed_model=embed_model,  # 임베딩 모델
            vectorstore_paths=vectorstore_paths,  # 소스별 경로
            unified_vectorstore_path=unified_vectorstore_path,  # 통합 경로
            sources=norm_sources,  # 선택 소스(체크박스 결과)
            force_rebuild=False,  # 강제는 아님(변경 있을 때만)
        )

    # 검색  # 실제로 벡터 검색을 수행합니다.
    contexts = _query_unified_with_source_filter(  # 통합 컬렉션에서 “선택 소스만” 필터링 검색
        persist_dir=unified_vectorstore_path,  # 통합 DB 폴더
        query_text=question,  # 질문
        top_k=top_k,  # 결과 개수
        embed_model=embed_model,  # 임베딩 모델
        norm_sources=norm_sources,  # 선택 소스
    )

    if not contexts:  # 검색 결과가 없으면
        return "관련 근거를 찾지 못했습니다. 데이터가 인덱싱되었는지 확인해주세요.", []  # 안내 메시지

    try:
        answer = _generate_answer(question, contexts, llm_model)  # LLM으로 최종 답변 생성
    except Exception as e:  # LLM 호출 중 오류가 나면
        answer = f"LLM 호출 중 오류가 발생했습니다: {e}\n\n아래는 검색된 근거입니다."  # 오류 메시지+근거 제공

    return answer, contexts  # 최종 답변 문자열 + 근거 리스트 반환


# streamlit_app.py에서 찾는 별칭 함수들  # 프론트엔드 코드가 query 대신 다른 이름으로 호출해도 동작하도록 “별칭”을 제공합니다.
def ask(question: str, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:  # ask()도 query()와 동일 동작
    return query(question=question, **kwargs)  # 결국 query를 호출

def run(question: str, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:  # run()도 query()와 동일 동작
    return query(question=question, **kwargs)  # 결국 query를 호출

def rag(question: str, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:  # rag()도 query()와 동일 동작
    return query(question=question, **kwargs)  # 결국 query를 호출

def answer(question: str, **kwargs) -> Tuple[str, List[Dict[str, Any]]]:  # answer()도 query()와 동일 동작
    return query(question=question, **kwargs)  # 결국 query를 호출
