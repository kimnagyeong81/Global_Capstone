from pipelines.upload_sorter import process_uploads

from pipelines.email_pipeline import build_email_index, ask_email
from pipelines.slack_pipeline import build_slack_index, ask_slack

from pipelines.voice_stt_pipeline import run_voice_stt
from pipelines.voice_rag_pipeline import build_voice_index, ask_voice

from pipelines.unified_rag_pipeline import build_unified_index, ask_unified


def main():
    print("""
================ Global Capstone Unified Runner ================

[데이터 전처리]
1. 업로드 분류 (Uploads → Email / Slack / Voice)
4. Voice STT + 화자분할

[개별 RAG 인덱싱]
2. Email 인덱싱
3. Slack 인덱싱
5. Voice 인덱싱

[개별 RAG 질의]
6. Email 질문
7. Slack 질문
8. Voice 질문

[통합 RAG]
9. Unified RAG 인덱싱 (Email + Slack + Voice)
10. Unified RAG 질문 (출처 포함)

===============================================================
""")

    m = input("선택: ").strip()

    if m == "1":
        process_uploads()

    elif m == "2":
        build_email_index()

    elif m == "3":
        build_slack_index()

    elif m == "4":
        run_voice_stt()

    elif m == "5":
        build_voice_index()

    elif m == "6":
        q = input("질문: ")
        print(ask_email(q))

    elif m == "7":
        q = input("질문: ")
        print(ask_slack(q))

    elif m == "8":
        q = input("질문: ")
        print(ask_voice(q))

    elif m == "9":
        build_unified_index()

    elif m == "10":
        q = input("질문: ")
        print(ask_unified(q))

    else:
        print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()

