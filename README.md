#  AI 면접관 시스템 (AI Interviewer Agent v2.0)

LangChain, OpenAI, LangGraph, ChromaDB 등을 활용하여 이력서 기반 질문 생성 → 답변 평가 → 자동 보고서 생성까지 자동화한 LLM 기반 인터뷰 시스템입니다.

---

##  주요 기능

| 기능 모듈 | 설명 |
|----------|------|
| **이력서 분석** (`analyze_resume`) | GPT-4o를 이용해 이력서를 요약하고, 핵심 키워드/강점/약점 키워드 추출 |
| **면접 스타일 설정** (`set_style`) | 구조화, 기술, 상황 기반 등 다양한 스타일 중 무작위 3개 선택 |
| **질문 생성 전략** (`generate_question_strategy`) | 스타일별 질문 전략 설계 및 예시 질문 구성 |
| **심층 질문 생성** (`generate_question`) | 기존 질문, 답변, 평가 내용을 반영한 심화 질문 생성 및 벡터 DB 저장 |
| **답변 평가** (`evaluate_answer`) | 관련성/구체성 기준으로 5점 척도 평가 및 사유 분석 |
| **자기 반성 평가** (`reflection`) | 최근 평가의 타당성을 자가 판단하여 흐름 분기 (`정상`/`재평가 필요`) |
| **흐름 제어** (`direction`) | 다음 진행 단계 결정: 질문, 평가, 리플렉션, 요약 등 |
| **면접 보고서 생성** (`summarize_interview`) | 전체 대화/평가/전략을 바탕으로 성향 분석 + 종합 피드백 리포트 출력 |

---

##  사용 기술

- **LangChain + LangGraph**: 인터뷰 흐름 구성 및 상태 분기
- **OpenAI GPT-4o-mini**: 질문/평가/요약 생성
- **ChromaDB**: 벡터 기반 RAG 구조 구현 (답변 히스토리 저장)
- **Gradio**: 프론트엔드 UI (외부 연동 시)

---
## 주의사항
- OpenAI API 키 필요 (ChatOpenAI 사용 시)
- Gradio UI 연동은 선택적 (함수 기반 CLI 실행 가능)
- ChromaDB는 collection_name="interview_qa"로 설정됨


