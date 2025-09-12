# IR-RAG 기반 대화형 검색·생성 시스템

## Team

| 이름  | 역할 |
|:---:|:--:|
| 김두환 | 팀장 |
| 이나경 | 팀원 |
| 조의영 | 팀원 |
| 박성진 | 팀원 |
| 편아현 | 팀원 |

## 0. Overview

이 프로젝트는 정보 검색(IR)과 생성형 모델을 결합한 RAG(Retrieval-Augmented Generation) 파이프라인을 구축하여, 사용자의 질의에 대해 관련 문서를 검색하고 이를 근거로 자연스러운 대화형
답변을 생성하는 것을 목표로 합니다. 대화 주제는 과학 상식으로 한정되며, 다중 턴 대화 시나리오를 지원합니다.

### Environment

- Python: >=3.11,<3.14 (권장: 3.12)
- 환경 관리: pyenv
- 실행 UI: Streamlit
- 프레임워크/라이브러리: LangChain, sentence-transformers 등
- 백엔드/스토리지(옵션): Chroma, Pinecone, Elasticsearch

설치 예시

```
bash
# pyenv로 프로젝트 파이썬 버전 설정
pyenv install -s 3.12.11
pyenv local 3.12.11

# 가상환경 생성 및 활성화 (원하는 방식 사용)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 필수 패키지 설치
pip install -e .

# 개발 도구(옵션) 설치
pip install -e ".[dev]"
```

### Requirements

- 환경변수 관리: python-dotenv
- 웹 UI: streamlit
- 체인/프롬프트/도구: langchain, langchain-core, langchain-community
- 문서 분할: langchain-text-splitters
- 백엔드 통합:
    - Embedding/VectorDB: langchain-chroma, langchain-pinecone
    - 검색엔진: langchain-elasticsearch
    - 모델 제공자: langchain-openai, langchain-upstage, langchain-ollama, langchain-huggingface, langchain-anthropic
- 임베딩: sentence-transformers
- 실험/추적: langsmith
- 개발 편의(옵션): ruff, pre-commit, jupyterlab, notebook, ipykernel, tqdm

## 1. Competition Info

이 프로젝트에서 진행하는 경진대회 정보는 첨부 문서를 참고하세요.

- 첨부: IR_competition_guide.md

### Overview

- 과학 상식 도메인에 대한 대화형 IR-RAG 시스템 구현
- 사용자의 질의 의도 분석 → 문서 검색/리랭킹 → 레퍼런스 기반 답변 생성
- 멀티턴 대화와 일반 대화(검색 불필요) 시나리오 포함

### Timeline

- 첨부 문서 및 운영 공지에 따름

## 2. Components

### Directory

- 초기 구조 예시(팀 내 합의에 따라 변경 가능)

```

├── src/
│   └── smart_llm/           # 앱/엔진 소스
├── app/                     # streamlit 등 프런트엔드 진입점
├── data/
│   ├── corpus/              # 원문/색인 대상 문서
│   └── eval/                # 평가용 대화 데이터
├── notebooks/               # 실험/EDA 노트북
├── configs/                 # 환경/체인/모델 설정
└── logs/                    # 런타임/평가 로그
```

## 3. Data Description

### Dataset overview

- 색인 대상: 교육/과학 상식 도메인 문서
- 형태: 문서 ID, 출처, 내용 등 필드 구성
- 평가 데이터: 과학 상식 질의 + 일반 대화 혼합의 멀티턴 메시지

### EDA

- 코퍼스 분포/길이 분석
- 질의 유형/난이도/오류 사례 수집
- 임베딩 분포 및 리트리버 성능 기초 점검

### Data Processing

- 텍스트 정제/정규화
- 문서 분할(Chunking) 전략 수립
- 메타데이터 설계 및 인덱싱

## 4. Modeling

### Model description

- RAG 파이프라인
    - Retriever: Sparse/Dense/Hybrid 선택
    - Reranker: LLM 또는 학습기반 리랭킹
    - Generator: LLM 기반 레퍼런스 반영 생성
- 선택 근거: 과학 상식 도메인 적합성, 멀티턴 대응, 비용/성능 균형

### Modeling Process

- 질의 의도 분석 → Standalone Query 생성
- 후보군 확장(Top-K) → 리랭킹/중복제거
- 컨텍스트 구성 → 답변 생성/인용 처리
- 오프라인 MAP 및 온라인 시뮬레이션 평가

## 5. Result

### Leader Board

- 추후 업데이트(최종 점수/순위/스크린샷)

### Presentation

- 추후 업데이트(발표자료 링크)


### Meeting Log

- 추후 업데이트(예: Notion/Google Docs)

### Reference

- 첨부 문서: IR_competition_guide.md
- 사용 라이브러리 공식 문서 및 벤치마크 레퍼런스
