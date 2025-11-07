# pipelines/slack/ask_slack.py
import sys, json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# JSON 파일 경로
DATA_PATH = Path(__file__).parent / "slack_messages.json"
EMB_MODEL = "BAAI/bge-m3"
TOP_K = 5

# 임베딩 모델 로드 (한 번만)
_emb = SentenceTransformer(EMB_MODEL)

# 데이터 불러오기
with open(DATA_PATH, "r", encoding="utf-8") as f:
    MESSAGES = json.load(f)

def search(query: str):
    """JSON Slack 데이터에서 쿼리와 가장 유사한 메시지 찾기"""
    qvec = _emb.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    docs = [m["text"] for m in MESSAGES]
    dvecs = _emb.encode(docs, convert_to_tensor=True, normalize_embeddings=True)

    # 코사인 유사도 계산
    scores = util.cos_sim(qvec, dvecs)[0]
    top_idx = scores.topk(k=TOP_K).indices.tolist()

    results = []
    for i in top_idx:
        msg = MESSAGES[i]
        score = float(scores[i])
        results.append((msg, score))
    return results

def format_result(msg, score):
    head = f"[{msg['channel_name']} | @{msg['user_name']}] ({msg['category']}, {msg['ts']})"
    body = msg['text']
    return f"{head}\n{body}\n(score={score:.3f})\n"

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "billing API 관련 질문"
    hits = search(query)
    print("\n=== Top Results ===")
    for msg, score in hits:
        print(format_result(msg, score))
