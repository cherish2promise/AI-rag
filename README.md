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


### 🔹 Step 3
```python
import pandas as pd
import numpy as np
import os
import ast
import fitz  # PyMuPDF
from docx import Document
import random
import openai
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Annotated, Literal, Sequence, TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
```
**설명:**
- 

### 🔹 Step 4
```python
def load_api_keys(filepath="api_key.txt"):
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

path = '/content/drive/MyDrive/aivle/project_genai/'
# API 키 로드 및 환경변수 설정
load_api_keys(path + 'api_key.txt')
```
**설명:**
- 

### 🔹 Step 5
```python
print(os.environ['OPENAI_API_KEY'][:30])
```
**설명:**
- 

### 🔹 Step 6
```python
def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    else:
        raise ValueError("지원하지 않는 파일 형식입니다. PDF 또는 DOCX만 허용됩니다.")
```
**설명:**
- 

### 🔹 Step 7
```python
file_path = path + 'Resume_sample.pdf'
```
**설명:**
- 

### 🔹 Step 8
```python
resume_text = extract_text_from_file(file_path)
resume_text
```
**설명:**
- 

### 🔹 Step 9
```python
from typing import TypedDict, List, Dict

class InterviewState(TypedDict):
    # 고정 정보
    resume_text: str
    resume_summary: str
    resume_keywords: List[str]
    question_strategy: Dict[str, Dict]
    style : List[str]

    # 인터뷰 로그
    current_question: str
    current_answer: str
    current_strategy: str
    conversation: List[Dict[str, str]]
    evaluation : List[Dict[str, int]]
    next_step : str

    #reflection
    reflection_status:str


    # 의견 피드백
    feedback : str
    tendency :str
    tot_opinion :str
```
**설명:**
- 

### 🔹 Step 10
```python
initial_state: InterviewState = {
    "resume_text": resume_text,
    "resume_summary": '',
    "resume_keywords": [],
    "question_strategy": {},
    "style" : [],
    "reflection_status" :'',

    "current_question": '',
    "current_answer": '',
    "current_strategy": '',
    "conversation": [],
    "evaluation": [],
    "next_step" : '',
    "feedback" : '',
    "tendency" : '',
    "tot_opinion":''
}

initial_state
```
**설명:**
- 

### 🔹 Step 11
```python
def analyze_resume(state: InterviewState) -> InterviewState:
    resume_text = state.get("resume_text", "")
    if not resume_text:
        raise ValueError("resume_text가 비어 있습니다. 먼저 텍스트를 추출해야 합니다.")

    llm = ChatOpenAI(model="gpt-4o-mini")

    # 요약 프롬프트 구성
    summary_prompt = ChatPromptTemplate.from_template(
        '''당신은 이력서를 바탕으로 인터뷰 질문을 설계하는 AI입니다.
        다음 이력서 및 자기소개서 내용에서 질문을 뽑기 위한 중요한 내용을 10문장 정도로 요약을 해줘(요약시 ** 기호는 사용하지 말것):\n\n{resume_text}'''
    )
    formatted_summary_prompt = summary_prompt.format(resume_text=resume_text)
    summary_response = llm.invoke(formatted_summary_prompt)
    resume_summary = summary_response.content.strip()

    # 키워드 추출 프롬프트 구성
    keyword_prompt = ChatPromptTemplate.from_template(
        '''당신은 이력서를 바탕으로 인터뷰 질문을 설계하는 AI입니다.
        다음 이력서 및 자기소개서내용에서 질문을 뽑기 위한 중요한 핵심 키워드를 5~10개 추출해줘.
        추가로 이력서를 보고 강점과 약점에 관련된 키워드를 각각 3개씩 추출해줘.
        도출한 핵심 키워드만 쉼표로 구분해줘:\n\n{resume_text}'''
    )
    formatted_keyword_prompt = keyword_prompt.format(resume_text=resume_text)
    keyword_response = llm.invoke(formatted_keyword_prompt)

    parser = CommaSeparatedListOutputParser()
    resume_keywords = parser.parse(keyword_response.content)

    return {
        **state,
        "resume_summary": resume_summary,
        "resume_keywords": resume_keywords,
    }
```
**설명:**
- 

### 🔹 Step 12
```python
import random
styler=["구조화 면접","비구조화 면접", "역량 기반 면접","상황 기반 면접","기술 면접","케이스 면접"]
style_set= random.sample(styler,3)
def set_style(state: InterviewState) -> InterviewState:
    # 여기에 코드를 완성합니다
    style=style_set
    return {
        **state,
        "style": style_set
    }
```
**설명:**
- 

### 🔹 Step 13
```python
i=0

def generate_question_strategy(state: InterviewState) -> InterviewState:
    global i
    # 여기에 코드를 완성합니다

    llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")

    prompt = PromptTemplate(
      input_variables=["resume_summary", "resume_keywords","style"],
      template = """
        이력서 요약: {resume_summary}
        키워드: {resume_keywords}
        준비된 이력서 요약 ,핵심 키워드 ,강점 키워드 ,약점 키워드를 참조해서 면접자에게 질문을 하려고 해

        면접 스타일은 {style}에 맞게끔 한개의 질문을 생성해줘
        """
    )

    chain = prompt | llm
    response = chain.invoke({
        "resume_summary": state["resume_summary"],
        "resume_keywords": ", ".join(state["resume_keywords"]),
        "style":state["style"][i]
    })

    strategy_dict = {}
    current_style = state["style"][i]

    strategy_dict[current_style] = {
        "예시 질문": [response.content.strip()]
    }

    i += 1

    # return 코드는 제공합니다.
    return {
        **state,
        "question_strategy": strategy_dict
    }
```
**설명:**
- 

### 🔹 Step 14
```python
def preProcessing_Interview(file_path: str) -> InterviewState:
    # 1. 텍스트 추출
    resume_text = extract_text_from_file(file_path)

    # 2. 초기 state 구성
    state: InterviewState = {
      "resume_text": resume_text,
      "resume_summary": '',
      "resume_keywords": [],
      "question_strategy": {},
      "style" : [],

      "current_question": '',
      "current_answer": '',
      "current_strategy": '',
      "conversation": [],
      "evaluation": [],
      "next_step" : '',
      "feedback" : '',
      "tendency" : '',
      "tot_opinion":'',
      "reflection_status":''
    }

    # 3. 이력서 분석
    state = analyze_resume(state)
    # style 추가
    state=set_style(state)
    # 4. 질문 전략 수립
    state = generate_question_strategy(state)

    # 5. 첫 질문 추출
    first_style = state["style"][0]
    strategy = state["question_strategy"].get(first_style, {})
    example_questions = strategy.get("예시 질문", [])

    selected_question = example_questions[0] if example_questions else "1분 자기소개를 해주세요"

    return {
        **state,
        "current_question": selected_question,
        "current_strategy": "첫 질문"
    }
```
**설명:**
- 

### 🔹 Step 15
```python
state=preProcessing_Interview(file_path)
state
```
**설명:**
- 

### 🔹 Step 16
```python
def update_current_answer(state: InterviewState) -> InterviewState:

    return {
        **state,
        "current_answer": state["current_answer"].strip()
    }
```
**설명:**
- 

### 🔹 Step 17
```python
def evaluate_answer(state: InterviewState) -> InterviewState:
    from typing import List, Dict
    import re

    llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")

    if state.get("reflection_status") == "정상":
        return state

    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """당신은 인사 전문가입니다. 면접자의 답변을 아래 항목에 따라 평가해 주세요.

각 항목마다 다음 정보를 제공해 주세요:
1. 점수: 5~1점 중 하나 (가장 높은 점수는 5점 가장 낮은 점수는 1점)
2. 사유: 그 이유를 서술형으로 설명

형식:
[질문과의 관련성]
점수: X
사유: ...

[답변의 구체성]
점수: X
사유: ...

질문: {question}
답변: {answer}
"""),
        ("user", "위 답변을 평가해 주세요.")
    ])

    chain = prompt | llm
    result = chain.invoke({
        "question": state["current_question"],
        "answer": state["current_answer"],
    })

    content = result.content.strip()

    # 응답 파싱
    evaluation: List[Dict[str, str]] = []
    current_item = ""

    score = None
    reason = ""

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("[질문과의 관련성]"):
            current_item = "질문과의 관련성"
        elif line.startswith("[답변의 구체성]"):
            current_item = "답변의 구체성"
        elif line.startswith("점수:"):
            score_match = re.search(r"점수:\s*(\d+)", line)
            if score_match:
                score = int(score_match.group(1))
        elif line.startswith("사유:"):
            reason = line.replace("사유:", "").strip()
            if current_item and score is not None:
                evaluation.append({
                    "항목": current_item,
                    "점수": score,
                    "사유": reason
                })
                # 초기화
                current_item = ""
                score = None
                reason = ""


    # 대화 기록
    conversation_entry = {
        "q": state["current_question"],
        "a": state["current_answer"]
    }

    return {
        **state,
        "evaluation": state["evaluation"] + evaluation,
        "conversation": state["conversation"] + [conversation_entry],
    }
```
**설명:**
- 

### 🔹 Step 18
```python
state=evaluate_answer(state)
state['evaluation']
```
**설명:**
- 

### 🔹 Step 19
```python
def reflection(state: InterviewState) -> InterviewState:
    llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")

    # 최근 평가 항목
    recent_eval = state["evaluation"][-2:] if len(state["evaluation"]) >= 2 else []
    eval_text = "\n".join([f"[{e['항목']}] 점수: {e['점수']}, 사유: {e['사유']}" for e in recent_eval])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 면접자입니다. 아래 질문과 답변, 그리고 평가 내용을 보고 해당 평가가 타당하다고 생각하는지 판단해주세요.

결과는 반드시 "정상" 또는 "재평가 필요" 중 하나로만 대답해주세요.

질문: {question}
답변: {answer}

평가:
{evaluation}
"""),
        ("user", "이 평가는 타당하다고 생각하나요?")
    ])

    chain = prompt | llm
    result = chain.invoke({
        "question": state["current_question"],
        "answer": state["current_answer"],
        "evaluation": eval_text
    })

    status = result.content.strip()
    if "재평가" in status:
        reflection_status = "재평가 필요"
        # 평가 초기화
        updated_evaluation = state["evaluation"][:-2]  # 최근 평가 제거
    else:
        reflection_status = "정상"
        updated_evaluation = state["evaluation"]  # 유지

    return {
        **state,
        "reflection_status": reflection_status,
        "evaluation": updated_evaluation
    }
```
**설명:**
- 

### 🔹 Step 20
```python
state = reflection(state)
state
```
**설명:**
- 

### 🔹 Step 21
```python
state=evaluate_answer(state)
state
```
**설명:**
- 

### 🔹 Step 22
```python
def direction(state: InterviewState) -> InterviewState:
    status = state.get("reflection_status", "").strip()

    if len(state["evaluation"])<3:
      if status == "정상":
          next_step = "generate_question"
      else:
          next_step = "reflection"
    else :
      next_step="summarize_interview"

    return {
        **state,
        "reflection_status" : '',
        "next_step": next_step
    }
```
**설명:**
- 

### 🔹 Step 23
```python
state['evaluation']
```
**설명:**
- 

### 🔹 Step 24
```python
state=direction(state)
state['next_step']
```
**설명:**
- 

### 🔹 Step 25
```python

```
**설명:**
- 

### 🔹 Step 26
```python
state=generate_question(state)
state
```
**설명:**
- 

### 🔹 Step 27
```python
state["current_answer"]="알고리즘을 잘 선정하지 못하겠습니다."
state=update_current_answer(state)
state=evaluate_answer(state)
```
**설명:**
- 

### 🔹 Step 28
```python
state
```
**설명:**
- 

### 🔹 Step 29
```python
def summarize_interview(state: InterviewState) -> InterviewState:
    llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")

    resume_text = state["resume_text"].strip()
    evaluation_text = "\n".join([
        f"{e['항목']} - 점수: {e['점수']}, 사유: {e['사유']}" for e in state["evaluation"]
    ])
    conversation_text = "\n".join([
        f"Q{i+1}: {qa['q']}\nA{i+1}: {qa['a']}" for i, qa in enumerate(state["conversation"])
    ])
    strategy_text = "\n".join([
        f"{k}: {', '.join(v['예시 질문'])}" for k, v in state["question_strategy"].items()
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 채용 담당자로, 다음 면접 내용을 기반으로 인터뷰 보고서를 작성해야 합니다.

[이력서]
{resume}

[면접 질문 및 답변]
{conversation}

[평가 내용]
{evaluation}

[질문 전략]
{strategy}

--- 작성 기준 ---
1. 성향 분석: 지원자의 대답을 기반으로 성격이나 커뮤니케이션 스타일, 업무 태도를 분석하세요.
2. 전체 의견: 질문 전략별 답변 스타일, 강점과 약점을 종합적으로 정리하세요.
""")
    ])

    chain = prompt | llm
    result = chain.invoke({
        "resume": resume_text,
        "conversation": conversation_text,
        "evaluation": evaluation_text,
        "strategy": strategy_text
    })

    content = result.content.strip()

    # 간단한 추출
    feedback, tendency, tot_opinion = "", "", ""
    lines = content.splitlines()
    current = None

    for line in content.splitlines():
      l = line.strip()
      if "성향 분석" in l:
        current = "tendency"
        continue
      elif "전체 의견" in l or "강점" in l:
        current = "tot_opinion"
        continue
      if current == "tendency":
        tendency += l + "\n"
      elif current == "tot_opinion":
        tot_opinion += l + "\n"

    # ✅ 최종 등급 계산
    scores = [e["점수"] for e in state["evaluation"] if isinstance(e["점수"], (int, float))]
    avg_score = sum(scores) / len(scores) if scores else 0

    if avg_score >= 4.5:
        feedback = "추천"
    elif avg_score >= 3.5:
        feedback = "조건부 추천"
    else:
        feedback = "비추천"

    # 리포트 출력
    report_lines = []
    report_lines.append("\n인터뷰 종료 리포트")
    report_lines.append("============================================================")
    report_lines.append("")

    report_lines.append("[1] 기본 정보")
    report_lines.append("지원자 성명 : 홍길동")  # 나중에 state에서 가져오도록 개선 가능
    report_lines.append("지원 직무 / 부서 : AI 개발팀")
    report_lines.append("")

    for i, qa in enumerate(state["conversation"]):
        report_lines.append(f"질문 {i+1}: {qa['q']}")
        report_lines.append(f"답변 {i+1}: {qa['a']}")
        report_lines.append("")

    report_lines.append("------------------------------------------------------------")
    report_lines.append("[2] 종합 평가")
    report_lines.append(f"추천 여부: {feedback}")
    report_lines.append("")

    report_lines.append("[3] 종합 의견 및 피드백")
    report_lines.append(f"성향 분석: {tendency}")
    report_lines.append("")
    report_lines.append("전체 의견:")
    report_lines.append(tot_opinion)
    report_lines.append("============================================================")

    print("\n".join(report_lines))

    return {
        **state,
        "feedback": feedback.strip(),
        "tendency": tendency.strip(),
        "tot_opinion": tot_opinion.strip()
    }
```
**설명:**
- 

### 🔹 Step 30
```python
state = summarize_interview(state)
state
```
**설명:**
- 

### 🔹 Step 31
```python
# 6) Agent --------------------
# 분기 판단 함수
# ─── graph 설정 부분 ───────────────────────────────
from langgraph.graph import StateGraph, END

graph = StateGraph(InterviewState)

# update_answer 노드 등록/연결을 모두 빼고,
# 진입점을 evaluate_answer 로 지정
graph.set_entry_point("evaluate_answer")

graph.add_node("evaluate_answer", evaluate_answer)
graph.add_node("reflection", reflection)
graph.add_node("direction", direction)
graph.add_node("generate_question", generate_question)
graph.add_node("summarize_interview", summarize_interview)

# evaluate_answer 이후로는 기존 분기 로직 그대로
graph.add_edge("evaluate_answer", "direction")
graph.add_conditional_edges("direction", lambda s: s["next_step"], {
    "reflection": "reflection",
    "generate_question": "generate_question",
    "summarize_interview": "summarize_interview",
})
graph.add_edge("reflection", "evaluate_answer")
graph.add_edge("generate_question", "evaluate_answer")  # 이 부분 간단히 넘겨도 됩니다.
graph.add_edge("summarize_interview", END)


# 최종 빌드
interview_app = graph.compile()
```
**설명:**
- 

### 🔹 Step 32
```python
# 파일 입력
file_path = path + 'Resume_sample.pdf'
i=0
state_s = preProcessing_Interview(file_path)
state_s
```
**설명:**
- 

### 🔹 Step 33
```python
# 사용자 응답 루프
state_s = interview_app.invoke(state_s)
```
**설명:**
- 


