import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import streamlit as st

# =========================
# Paths (project structure)
# =========================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "Data")
PIPELINES_DIR = os.path.join(ROOT_DIR, "pipelines")
VECTORSTORES_DIR = os.path.join(ROOT_DIR, "vectorstores")

DEFAULT_VS = {
    "slack": "slack_bge_m3",
    "email": "emails_bge_m3",
    "voice": "voices_bge_m3",
    "unified": "unified_rag",
}

SOURCE_DIRS = {
    "email": os.path.join(DATA_DIR, "Emails"),
    "slack": os.path.join(DATA_DIR, "Slack"),
    "voice": os.path.join(DATA_DIR, "Voice"),
}

DATE_RANGE_OPTIONS = ["Today", "Past Week", "Past Month", "Past Quarter", "All Time"]
DEPARTMENTS = ["Engineering", "Product", "Marketing", "Sales", "HR"]

DOCS_DIR = os.path.join(DATA_DIR, "Documents")  # optional
SOURCE_DIRS_EXT = {
    "documents": DOCS_DIR,
    "email": SOURCE_DIRS["email"],
    "slack": SOURCE_DIRS["slack"],
    "voice": SOURCE_DIRS["voice"],
}

# =========================
# Page
# =========================
st.set_page_config(page_title="Global Capstone RAG Chat", layout="wide")

st.markdown(
    """
<style>
/* ---- Global spacing (title cut / top padding fix) ---- */
main .block-container{
  padding-top: 1.0rem !important;
  padding-bottom: 1.0rem !important;
}

/* ---- 3-panel layout helpers ---- */
.panel-card{
  background: rgba(255,255,255,1);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 14px 14px;
}

/* ---- Chat area ---- */
.chat-wrap {
  height: calc(100vh - 260px); /* header + options + sticky input 고려 */
  overflow-y: auto;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(0,0,0,0.06);
  background: #FFFFFF;
}

/* message rows */
.msg-row { display: flex; margin: 10px 0; }
.msg-left { justify-content: flex-start; }
.msg-right { justify-content: flex-end; }

/* bubble */
.bubble {
  max-width: 78%;
  padding: 12px 14px;
  border-radius: 16px;
  font-size: 0.95rem;
  line-height: 1.45;
  word-break: break-word;
  box-shadow: 0 1px 0 rgba(0,0,0,0.03);
}
.bubble-ai { background: #F2F3F5; color: #111827; border-top-left-radius: 6px; }
.bubble-user { background: #2563EB; color: #FFFFFF; border-top-right-radius: 6px; }

/* ---- Sticky input (ONLY inside center column) ---- */
.sticky-input {
  position: sticky;
  bottom: 0;
  background: #FFFFFF;
  padding-top: 12px;
  padding-bottom: 10px;
  border-top: 1px solid rgba(0,0,0,0.08);
  z-index: 50;
}

/* Textarea sizing */
.chat-textarea textarea {
  min-height: 62px !important;
  border-radius: 14px !important;
  padding-top: 14px !important;
  padding-right: 54px !important;
}

/* Send 버튼: Enterprise Navy */
div[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(180deg, #1F2A44 0%, #111827 100%) !important;
  color: #FFFFFF !important;
  border: 1px solid rgba(255,255,255,0.10) !important;

  width: 44px !important;
  height: 44px !important;
  min-width: 44px !important;
  min-height: 44px !important;

  border-radius: 14px !important;
  padding: 0 !important;

  display: flex !important;
  align-items: center !important;
  justify-content: center !important;

  box-shadow: 0 10px 24px rgba(17,24,39,0.18) !important;
  transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease !important;
}

div[data-testid="stFormSubmitButton"] button:hover {
  filter: brightness(1.06) !important;
  box-shadow: 0 14px 30px rgba(17,24,39,0.24) !important;
  transform: translateY(-1px) !important;
}

div[data-testid="stFormSubmitButton"] button:active {
  transform: translateY(0px) !important;
  box-shadow: 0 8px 18px rgba(17,24,39,0.18) !important;
}

/* hover */
.send-btn div.stButton > button:hover {
  filter: brightness(1.05) !important;
  box-shadow: 0 14px 30px rgba(17,24,39,0.24) !important;
  transform: translateY(-1px) !important;
}

/* active */
.send-btn div.stButton > button:active {
  transform: translateY(0px) !important;
  box-shadow: 0 8px 18px rgba(17,24,39,0.18) !important;
}

/* focus ring (키보드 탭 이동 시) */
.send-btn div.stButton > button:focus-visible {
  outline: none !important;
  box-shadow: 0 0 0 4px rgba(59,130,246,0.25), 0 10px 24px rgba(17,24,39,0.18) !important;
}

/* Left panel small title */
.sb-title {
  font-size: 0.95rem;
  font-weight: 800;
  margin: 0.2rem 0 0.6rem 0;
}
.sb-divider { margin: 0.85rem 0; opacity: 0.25; }
.sb-count { font-size: 0.78rem; opacity: 0.65; min-width: 3.2rem; text-align: right; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("RAG-based Enterprise Decision-Support AI Agent")
st.caption("Slack / Emails / Voice에서 맥락을 찾아 답변하는 AI")

# =========================
# Data model
# =========================
@dataclass
class RetrievedDoc:
    source: str
    title: str
    snippet: str
    metadata: Dict[str, Any]

# =========================
# Helpers
# =========================
def safe_import(module_path: str):
    try:
        mod = __import__(module_path, fromlist=["*"])
        return mod, None
    except Exception as e:
        return None, e

@st.cache_resource
def load_unified_pipeline():
    mod, err = safe_import("pipelines.unified_rag_pipeline")
    return mod, err

def pick_vectorstore_path(source: str, embed_tag: str) -> str:
    if embed_tag == "bge_m3":
        folder = DEFAULT_VS.get(source, DEFAULT_VS["unified"])
    else:
        if source == "email":
            folder = f"emails_{embed_tag}"
        elif source == "voice":
            folder = f"voices_{embed_tag}"
        elif source == "slack":
            folder = f"slack_{embed_tag}"
        else:
            folder = f"unified_{embed_tag}"
    return os.path.join(VECTORSTORES_DIR, folder)

def count_files_recursive(path: str) -> int:
    if not os.path.exists(path):
        return 0
    cnt = 0
    for root, _, files in os.walk(path):
        cnt += len([f for f in files if not f.startswith(".")])
    return cnt

@st.cache_data(show_spinner=False)
def get_source_counts_ext() -> Dict[str, int]:
    return {k: count_files_recursive(v) for k, v in SOURCE_DIRS_EXT.items()}

def normalize_contexts(contexts: Any) -> List[RetrievedDoc]:
    docs: List[RetrievedDoc] = []
    if contexts is None:
        return docs

    if isinstance(contexts, dict):
        for key in ["contexts", "documents", "docs", "sources", "results"]:
            if key in contexts and isinstance(contexts[key], list):
                contexts = contexts[key]
                break

    if not isinstance(contexts, list):
        return docs

    for c in contexts:
        if isinstance(c, dict):
            docs.append(
                RetrievedDoc(
                    source=str(c.get("source", c.get("type", "unknown"))),
                    title=str(c.get("title", c.get("subject", c.get("channel", "context")))),
                    snippet=str(c.get("snippet", c.get("text", c.get("content", ""))))[:1200],
                    metadata=dict(c.get("metadata", {})),
                )
            )
        else:
            docs.append(RetrievedDoc(source="unknown", title="context", snippet=str(c)[:1200], metadata={}))
    return docs

def call_unified_pipeline(
    pipeline_mod,
    question: str,
    sources: List[str],
    top_k: int,
    embed_tag: str,
    llm_model: str,
    date_range: str,
    departments: List[str],
) -> Tuple[str, List[RetrievedDoc], str]:
    candidate_fns = ["query", "run", "ask", "rag", "answer"]

    for fn_name in candidate_fns:
        if not hasattr(pipeline_mod, fn_name):
            continue

        fn = getattr(pipeline_mod, fn_name)
        vs_paths = {s: pick_vectorstore_path(s, embed_tag) for s in sources}
        unified_vs_path = pick_vectorstore_path("unified", embed_tag)

        try:
            out = fn(
                question=question,
                sources=sources,
                top_k=top_k,
                llm_model=llm_model,
                vectorstore_paths=vs_paths,
                unified_vectorstore_path=unified_vs_path,
                date_range=date_range,
                departments=departments,
            )
            if isinstance(out, tuple) and len(out) >= 2:
                return str(out[0]), normalize_contexts(out[1]), f"used: {fn_name}(keyword args)"
            if isinstance(out, dict):
                answer = str(out.get("answer", out.get("output", out.get("response", ""))))
                docs = normalize_contexts(out.get("contexts", out))
                return answer, docs, f"used: {fn_name}(dict output)"
            return str(out), [], f"used: {fn_name}(answer only)"
        except TypeError:
            try:
                out = fn(question=question, sources=sources, top_k=top_k)
                if isinstance(out, tuple) and len(out) >= 2:
                    return str(out[0]), normalize_contexts(out[1]), f"used: {fn_name}(minimal)"
                if isinstance(out, dict):
                    answer = str(out.get("answer", out.get("output", out.get("response", ""))))
                    docs = normalize_contexts(out.get("contexts", out))
                    return answer, docs, f"used: {fn_name}(dict minimal)"
                return str(out), [], f"used: {fn_name}(answer only minimal)"
            except Exception:
                continue
        except Exception:
            continue

    return (
        "unified_rag_pipeline에서 호출 가능한 함수(query/run/ask/rag/answer)를 찾지 못했어요.\n"
        "또는 함수 시그니처가 달라서 호출에 실패했어요.\n\n"
        "해결: pipelines/unified_rag_pipeline.py 안에 `query()` 함수 하나만 만들어서\n"
        "`question, sources, top_k`(+ 선택적으로 date_range, departments, llm_model)를 받게 해주세요.",
        [],
        "error: no callable function matched",
    )

# =========================
# Pipeline load status
# =========================
pipeline_mod, pipeline_err = load_unified_pipeline()
if pipeline_err:
    st.warning(f"⚠️ pipelines/unified_rag_pipeline.py import 실패: {pipeline_err}")

# =========================
# Session state
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 질문을 입력하면 RAG로 근거를 찾아 답변할게요."}]
if "last_contexts" not in st.session_state:
    st.session_state.last_contexts = []
if "debug_info" not in st.session_state:
    st.session_state.debug_info = ""

# =========================
# 3-panel Layout: Left : Middle : Right = 2 : 5 : 3
# =========================
left_col, chat_col, ctx_col = st.columns([2, 5, 3], gap="large")
counts = get_source_counts_ext()

# -------------------------
# Left panel (filters)
# -------------------------
with left_col:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)

    st.markdown("<div class='sb-title'>Source Types</div>", unsafe_allow_html=True)

    if "src_email" not in st.session_state:
        st.session_state.src_email = True
    if "src_slack" not in st.session_state:
        st.session_state.src_slack = True
    if "src_voice" not in st.session_state:
        st.session_state.src_voice = False

    def source_row(key: str, icon: str, label: str, count: int, default: bool):
        c1, c2 = st.columns([0.80, 0.20], vertical_alignment="center")
        with c1:
            st.session_state[key] = st.checkbox(f"{icon}  {label}", value=st.session_state.get(key, default))
        with c2:
            st.markdown(f"<div class='sb-count'>{count:,}</div>", unsafe_allow_html=True)

    source_row("src_email", "✉️", "Emails", counts.get("email", 0), True)
    source_row("src_slack", "💬", "Slack Messages", counts.get("slack", 0), True)
    source_row("src_voice", "🎙️", "Voice Transcripts", counts.get("voice", 0), False)

    st.markdown("<hr class='sb-divider'/>", unsafe_allow_html=True)

    selected_sources = []
    if st.session_state.src_email:
        selected_sources.append("email")
    if st.session_state.src_slack:
        selected_sources.append("slack")
    if st.session_state.src_voice:
        selected_sources.append("voice")

    st.markdown("<div class='sb-title'>Date Range</div>", unsafe_allow_html=True)
    date_range = st.radio(
        label="Date Range",
        options=DATE_RANGE_OPTIONS,
        index=4,
        label_visibility="collapsed",
        format_func=lambda x: f"📅  {x}",
    )

    st.markdown("<hr class='sb-divider'/>", unsafe_allow_html=True)

    st.markdown("<div class='sb-title'>Departments</div>", unsafe_allow_html=True)
    if "dept_selected" not in st.session_state:
        st.session_state.dept_selected = {d: False for d in DEPARTMENTS}
    for d in DEPARTMENTS:
        st.session_state.dept_selected[d] = st.checkbox(f"👥  {d}", value=st.session_state.dept_selected.get(d, False))
    departments = [d for d, v in st.session_state.dept_selected.items() if v]

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Middle panel (chat)
# -------------------------
with chat_col:
    with st.expander("Chat options", expanded=False):
        top_k = st.slider("Top-K contexts", 1, 20, 5)
        llm_model = st.text_input("LLM model name", value="qwen2.5:7b-instruct")
        embed_tag = st.text_input("Embedding tag (vectorstore folder suffix)", value="bge_m3")

    # Messages
    # --- Messages render (✅ 한 번에 렌더링해서 chat-wrap 안에 넣기) ---
    messages_html = []

    for m in st.session_state.messages:
        role = m.get("role", "assistant")
        content = (m.get("content", "") or "").replace("\n", "<br/>")

        if role == "user":
            messages_html.append(
                f"""
<div class="msg-row msg-right">
  <div class="bubble bubble-user">{content}</div>
</div>
"""
        )
        else:
            messages_html.append(
                f"""
<div class="msg-row msg-left">
  <div class="bubble bubble-ai">{content}</div>
</div>
"""
        )

    st.markdown(
        f"""
<div class='chat-wrap' id='chatWrap'>
       {''.join(messages_html)}
</div>
""",
        unsafe_allow_html=True,
)


    # Sticky input (inside center column)
    st.markdown("<div class='sticky-input'>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([0.92, 0.08], vertical_alignment="bottom")

        with c1:
            st.markdown("<div class='chat-textarea'>", unsafe_allow_html=True)
            user_q = st.text_area(
                "Ask",
                placeholder="Ask a question about your internal knowledge base…",
                label_visibility="collapsed",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='send-btn'>", unsafe_allow_html=True)
            send = st.form_submit_button("↑", type="secondary", use_container_width=False)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # On send
    if send:
        cleaned = (user_q or "").strip()
        if cleaned:
            st.session_state.messages.append({"role": "user", "content": cleaned})

            t0 = time.time()
            if pipeline_mod is None:
                answer = (
                    "unified_rag_pipeline.py를 불러오지 못해서 RAG 실행을 못했어요.\n"
                    "상단 경고(import error)를 먼저 해결해야 해요."
                )
                docs = []
                debug = "pipeline_mod is None"
            else:
                answer, docs, debug = call_unified_pipeline(
                    pipeline_mod=pipeline_mod,
                    question=cleaned,
                    sources=selected_sources,
                    top_k=top_k,
                    embed_tag=embed_tag,
                    llm_model=llm_model,
                    date_range=date_range,
                    departments=departments,
                )

            dt = time.time() - t0
            st.session_state.debug_info = (
                f"{debug} | {dt:.2f}s | sources={selected_sources} | top_k={top_k} "
                f"| date_range={date_range} | departments={departments}"
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.last_contexts = docs
            st.rerun()
        else:
            st.warning("질문을 입력해줘.")

# -------------------------
# Right panel (contexts)
# -------------------------
with ctx_col:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.subheader("Retrieved Contexts")

    docs: List[RetrievedDoc] = st.session_state.last_contexts or []
    if not docs:
        st.info("아직 검색된 근거가 없어요. 중앙에서 질문을 보내면 여기에 뜹니다.")
    else:
        for i, d in enumerate(docs, start=1):
            with st.expander(f"{i}. [{d.source}] {d.title}", expanded=(i <= 2)):
                st.write(d.snippet)
                if d.metadata:
                    st.caption("metadata")
                    st.json(d.metadata)

    st.divider()
    st.caption("debug")
    st.code(st.session_state.debug_info or "(none)")
    st.markdown("</div>", unsafe_allow_html=True)
