from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# RAG 처리 함수
from pipelines.voice.voice_reply_gen import ask_question


app = FastAPI()


# ======================================
# CORS 해결 (Swagger, React, HTML 모두 허용)
# ======================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 개발 단계에서는 전체 허용 OK
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================
# 요청 Body 정의
# ======================================
class Query(BaseModel):
    question: str


# ======================================
# RAG API 엔드포인트
# ======================================
@app.post("/api/chat")
def chat(query: Query):
    try:
        answer = ask_question(query.question)   # 문자열 전달
        return {"reply": answer}

    except Exception as e:
        # 서버 오류 발생 시 500 반환
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# Health Check 엔드포인트
# ======================================
@app.get("/")
def home():
    return {"status": "ok", "message": "FastAPI Voice RAG API running"}

