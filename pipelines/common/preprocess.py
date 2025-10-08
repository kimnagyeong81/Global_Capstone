# pipelines/common/preprocess.py
# 한 파일로 Slack / Email / Voice(STT) 전처리를 모두 지원
# 사용 예:
#   from pipelines.common.preprocess import clean_slack, clean_email, clean_voice, normalize_whitespace

import re
from typing import Iterable

# ───────────────────── 공통 정규식/도우미 ─────────────────────
RE_URL_ANGLE   = re.compile(r"<(https?://[^>|]+)(\|[^>]+)?>")   # <url|label> → url
RE_URL_PARENS  = re.compile(r"\((https?://[^)]+)\)")            # (url) → url
RE_INLINE_CODE = re.compile(r"`([^`]+)`")                       # `code` → code
RE_CODE_BLOCK  = re.compile(r"```.*?```", re.S)                 # ``` ... ```
RE_EMOJI       = re.compile(r":[a-z0-9_+\-]+:", re.I)           # :smile:
RE_SPACES      = re.compile(r"\s+")
RE_EMAIL_PII   = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
RE_PHONE_PII   = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{2,4}[-.\s]?){2,4}\d\b")

def mask_pii(text: str) -> str:
    """이메일/전화번호를 마스킹."""
    text = RE_EMAIL_PII.sub("[email]", text)
    text = RE_PHONE_PII.sub("[phone]", text)
    return text

def normalize_whitespace(text: str) -> str:
    """공백을 하나로 합치고 양끝을 정리."""
    return RE_SPACES.sub(" ", text).strip()

def common_clean(text: str, *, keep_code_block_placeholder: bool = True, pii: bool = True) -> str:
    """모든 소스에서 공통으로 적용할 기본 정리."""
    t = str(text)
    # 링크 표기 통일
    t = RE_URL_ANGLE.sub(r"\1", t)
    t = RE_URL_PARENS.sub(r"\1", t)
    # 코드 정리
    t = RE_INLINE_CODE.sub(r"\1", t)
    if keep_code_block_placeholder:
        t = RE_CODE_BLOCK.sub(" [code-block] ", t)
    else:
        t = RE_CODE_BLOCK.sub(" ", t)
    # 이모지/공백
    t = RE_EMOJI.sub("", t)
    t = normalize_whitespace(t)
    # 개인정보
    return mask_pii(t) if pii else t

# ───────────────────── Slack 전용 ─────────────────────
RE_MENTION = re.compile(r"<@([A-Z0-9]+)>")  # <@U123AB> → @U123AB
RE_CHAN    = re.compile(r"<#([A-Z0-9]+)\|[^>]+>")  # <#C123|channel-name> → #C123

def clean_slack(text: str, *, pii: bool = True) -> str:
    """슬랙 특유의 멘션/채널/링크/코드/이모지/공백/PII 정리."""
    t = str(text)
    t = RE_MENTION.sub(r"@\1", t)
    t = RE_CHAN.sub(r"#\1", t)
    t = common_clean(t, keep_code_block_placeholder=True, pii=pii)
    return t

# ───────────────────── Email 전용 ─────────────────────
RE_HTML_TAG  = re.compile(r"<[^>]+>")                   # 아주 가벼운 HTML 제거
RE_QUOTED    = re.compile(r"(?m)^(>+).*$")              # 인용 줄 제거
RE_SIG_LINE  = re.compile(r"(?m)^--\s*$")               # 서명 구분선
RE_FW_RE     = re.compile(r"(?i)\b(?:re|fw|fwd):\s*")   # 제목 접두어 정리

def strip_signature(text: str) -> str:
    """서명 구분선(--) 이후 잘라내기."""
    parts = RE_SIG_LINE.split(text, maxsplit=1)
    return parts[0] if parts else text

def clean_email(text: str, *, pii: bool = True, drop_quotes: bool = True, drop_signature: bool = True,
                remove_html: bool = True, normalize_subject_prefix: bool = False) -> str:
    """
    이메일 본문/제목 모두에 사용 가능.
    - HTML 태그 제거
    - 인용(>) 줄 제거
    - 서명(-- 이후) 제거
    - 공통 클린 + PII 마스킹
    """
    t = str(text)
    if remove_html:
        t = RE_HTML_TAG.sub(" ", t)
    if drop_quotes:
        t = RE_QUOTED.sub("", t)
    if drop_signature:
        t = strip_signature(t)
    t = common_clean(t, keep_code_block_placeholder=False, pii=pii)
    if normalize_subject_prefix:
        t = RE_FW_RE.sub("", t)  # 제목에서 RE:, FW: 제거용(원하면 사용)
    return t

# ───────────────────── Voice(STT) 전용 ─────────────────────
# 타임스탬프, 군말 제거 (한국어 중심. 필요시 리스트 확장)
RE_TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")  # 0:12 or 00:12:34
FILLERS_KO: Iterable[str] = ("음", "어", "그", "막", "뭐지", "약간", "그러니까", "어..", "음..")
FILLERS_EN: Iterable[str] = ("uh", "um", "you know", "like", "sort of", "kind of")

def remove_fillers(text: str, fillers: Iterable[str]) -> str:
    t = text
    for f in fillers:
        # 단어 경계 기준 제거(문장 속 의미 단어는 보존)
        t = re.sub(rf"(?<!\w){re.escape(f)}(?!\w)", " ", t, flags=re.I)
    return t

def clean_voice(text: str, *, pii: bool = True, drop_timestamps: bool = True, drop_fillers: bool = True) -> str:
    """STT 전사 텍스트 정리: 타임스탬프/군말/URL/이모지/PII/공백."""
    t = str(text)
    if drop_timestamps:
        t = RE_TIMESTAMP.sub(" ", t)
    if drop_fillers:
        t = remove_fillers(t, FILLERS_KO)
        t = remove_fillers(t, FILLERS_EN)
    t = common_clean(t, keep_code_block_placeholder=False, pii=pii)
    return t
