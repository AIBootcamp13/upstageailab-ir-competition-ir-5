# LangChain을 활용한 IR 경진대회 Submission 가이드

## 🎯 개요

LangChain을 사용하여 vector store를 구축하고 문서를 검색하여 submission.csv를 생성하는 완전한 파이프라인을 구현합니다.

## 📋 목차

1. [환경 설정](#환경-설정)
2. [데이터 준비 및 Vector Store 구축](#데이터-준비-및-vector-store-구축)
3. [검색 시스템 구현](#검색-시스템-구현)
4. [Submission 파일 생성](#submission-파일-생성)
5. [고급 최적화 기법](#고급-최적화-기법)

---

## 🛠️ 환경 설정

### 필요한 패키지 설치

```bash
pip install langchain langchain-openai langchain-community
pip install chromadb faiss-cpu  # vector store 옵션
pip install pandas numpy
```

### 기본 import 및 설정

```python
import pandas as pd
import json
import os
from typing import List, Dict, Optional

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# 환경 변수 설정
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

---

## 📚 데이터 준비 및 Vector Store 구축

### 1. 문서 데이터 로딩

```python
class DocumentLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_documents(self) -> List[Document]:
        """문서 데이터를 LangChain Document 형태로 로딩"""
        # CSV 또는 JSON 형태의 문서 데이터 로딩
        df = pd.read_csv(self.data_path)

        documents = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=row['content'],
                metadata={
                    'doc_id': row['doc_id'],
                    'source': row['source']
                }
            )
            documents.append(doc)

        print(f"총 {len(documents)}개 문서 로딩 완료")
        return documents


# 사용 예시
loader = DocumentLoader("documents.csv")
documents = loader.load_documents()
```

### 2. 문서 분할 (Chunking)

```python
class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """문서를 청크 단위로 분할"""
        chunked_docs = []

        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)

            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    }
                )
                chunked_docs.append(chunked_doc)

        print(f"총 {len(chunked_docs)}개 청크 생성")
        return chunked_docs


# 사용 예시
chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
chunked_documents = chunker.chunk_documents(documents)
```

### 3. Vector Store 구축

```python
class VectorStoreBuilder:
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

    def build_chroma_store(self, documents: List[Document],
                           persist_directory: str = "./chroma_db") -> Chroma:
        """ChromaDB를 사용한 Vector Store 구축"""
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name="science_docs"
        )

        print(f"ChromaDB Vector Store 구축 완료: {len(documents)}개 문서")
        return vectorstore

    def build_faiss_store(self, documents: List[Document]) -> FAISS:
        """FAISS를 사용한 Vector Store 구축"""
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        print(f"FAISS Vector Store 구축 완료: {len(documents)}개 문서")
        return vectorstore

    def load_existing_store(self, persist_directory: str) -> Chroma:
        """기존 Vector Store 로딩"""
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="science_docs"
        )
        return vectorstore


# 사용 예시
builder = VectorStoreBuilder()

# 새로 구축
vectorstore = builder.build_chroma_store(chunked_documents)

# 또는 기존 것 로딩
# vectorstore = builder.load_existing_store("./chroma_db")
```

---

## 🔍 검색 시스템 구현

### 1. 질의 분석 시스템

```python
class QueryAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 질의 분석 프롬프트
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 사용자의 질문을 분석하는 전문가입니다.

주어진 질문을 분석하여 다음을 판단해주세요:
1. 과학 상식 관련 질문인지 여부
2. 검색이 필요한 질문인지 여부

과학 상식 질문의 예: "광합성은 어떻게 일어나나요?", "중력의 원리가 뭐야?"
일상 대화의 예: "고마워", "잘 지내?", "네가 대답을 잘해줘서 신나"

응답 형식:
{{
    "is_science_query": true/false,
    "needs_search": true/false,
    "reasoning": "판단 근거"
}}"""),
            ("user", "{query}")
        ])

        # Standalone 쿼리 생성 프롬프트
        self.standalone_prompt = ChatPromptTemplate.from_messages([
            ("system", """주어진 대화 맥락과 후속 질문을 바탕으로, 문서 검색에 적합한 독립적인 쿼리를 생성해주세요.

생성 규칙:
1. 대화 맥락을 참고하여 완전한 질문으로 변환
2. 검색에 최적화된 키워드 포함
3. 한국어로 작성
4. 간결하고 명확하게

예시:
입력: "그것의 원리가 뭐야?" (앞선 대화에서 "광합성"에 대해 논의)
출력: "광합성의 원리와 과정"
"""),
            ("user", "대화 맥락: {chat_history}\n현재 질문: {question}")
        ])

    def analyze_query(self, query: str) -> Dict:
        """질의 분석 수행"""
        chain = self.analysis_prompt | self.llm

        try:
            response = chain.invoke({"query": query})
            result = json.loads(response.content)
            return result
        except:
            # 파싱 실패 시 기본값
            return {
                "is_science_query": True,
                "needs_search": True,
                "reasoning": "분석 실패로 기본값 적용"
            }

    def generate_standalone_query(self, question: str,
                                  chat_history: str = "") -> str:
        """검색용 독립 쿼리 생성"""
        chain = self.standalone_prompt | self.llm

        response = chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

        return response.content.strip()


# 사용 예시
analyzer = QueryAnalyzer()

query = "기억상실증의 원인이 뭐야?"
analysis = analyzer.analyze_query(query)
print(f"분석 결과: {analysis}")

if analysis["needs_search"]:
    standalone_query = analyzer.generate_standalone_query(query)
    print(f"검색 쿼리: {standalone_query}")
```

### 2. 검색 시스템

```python
class IRSearchSystem:
    def __init__(self, vectorstore, top_k: int = 10):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        self.analyzer = QueryAnalyzer()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def search(self, query: str, chat_history: str = "") -> List[str]:
        """주어진 쿼리로 문서 검색"""

        # 1. 질의 분석
        analysis = self.analyzer.analyze_query(query)

        if not analysis["needs_search"]:
            return []  # 검색 불필요한 경우 빈 리스트 반환

        # 2. Standalone 쿼리 생성
        standalone_query = self.analyzer.generate_standalone_query(
            query, chat_history
        )

        # 3. 문서 검색
        retrieved_docs = self.retriever.invoke(standalone_query)

        # 4. 문서 ID 추출 (중복 제거)
        doc_ids = []
        seen_ids = set()

        for doc in retrieved_docs:
            doc_id = doc.metadata.get('doc_id')
            if doc_id and doc_id not in seen_ids:
                doc_ids.append(doc_id)
                seen_ids.add(doc_id)

        return doc_ids[:3]  # Top-3 반환

    def search_with_reranking(self, query: str,
                              chat_history: str = "") -> List[str]:
        """LLM 리랭킹을 포함한 고급 검색"""

        # 1. 기본 검색으로 더 많은 후보 추출
        analysis = self.analyzer.analyze_query(query)

        if not analysis["needs_search"]:
            return []

        standalone_query = self.analyzer.generate_standalone_query(
            query, chat_history
        )

        # 더 많은 후보 추출 (10개)
        retrieved_docs = self.vectorstore.similarity_search(
            standalone_query, k=10
        )

        if not retrieved_docs:
            return []

        # 2. LLM을 사용한 리랭킹
        reranked_docs = self._rerank_with_llm(query, retrieved_docs)

        # 3. Top-3 문서 ID 반환
        doc_ids = []
        seen_ids = set()

        for doc in reranked_docs[:3]:
            doc_id = doc.metadata.get('doc_id')
            if doc_id and doc_id not in seen_ids:
                doc_ids.append(doc_id)
                seen_ids.add(doc_id)

        return doc_ids

    def _rerank_with_llm(self, query: str,
                         documents: List[Document]) -> List[Document]:
        """LLM을 사용한 문서 리랭킹"""

        rerank_prompt = ChatPromptTemplate.from_messages([
            ("system", """주어진 질문과 문서들을 보고, 질문에 대한 답변에 가장 유용한 순서로 문서를 정렬해주세요.

응답 형식: 문서 인덱스를 관련성이 높은 순서대로 나열 (예: [2, 0, 1, 4, 3])
"""),
            ("user", """질문: {query}

문서들:
{documents}

관련성이 높은 순서로 문서 인덱스를 정렬해주세요:""")
        ])

        # 문서 텍스트 준비
        doc_texts = []
        for i, doc in enumerate(documents):
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            doc_texts.append(f"문서 {i}: {content}")

        doc_string = "\n\n".join(doc_texts)

        try:
            chain = rerank_prompt | self.llm
            response = chain.invoke({
                "query": query,
                "documents": doc_string
            })

            # 순서 파싱
            order_str = response.content.strip()
            order = eval(order_str)  # 주의: 실제 프로덕션에서는 더 안전한 파싱 필요

            return [documents[i] for i in order if i < len(documents)]

        except:
            # 리랭킹 실패 시 원본 순서 반환
            return documents


# 사용 예시
search_system = IRSearchSystem(vectorstore, top_k=10)

# 기본 검색
query = "광합성에서 산소는 어떻게 생성되나요?"
doc_ids = search_system.search(query)
print(f"검색 결과: {doc_ids}")

# 리랭킹 포함 검색
doc_ids_reranked = search_system.search_with_reranking(query)
print(f"리랭킹 결과: {doc_ids_reranked}")
```

---

## 📤 Submission 파일 생성

### 1. 평가 데이터 처리

```python
class SubmissionGenerator:
    def __init__(self, search_system: IRSearchSystem):
        self.search_system = search_system

    def load_evaluation_data(self, eval_path: str) -> Dict:
        """평가 데이터 로딩"""
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        print(f"평가 데이터 로딩: {len(eval_data)}개 쿼리")
        return eval_data

    def extract_final_query(self, messages: List[Dict]) -> str:
        """멀티턴 대화에서 최종 사용자 질문 추출"""
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        if user_messages:
            return user_messages[-1]['content']
        return ""

    def extract_chat_history(self, messages: List[Dict]) -> str:
        """대화 이력 추출"""
        history = []
        for msg in messages[:-1]:  # 마지막 메시지 제외
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            history.append(f"{role}: {content}")

        return "\n".join(history)

    def generate_submission(self, eval_data: Dict,
                            output_path: str = "submission.csv"):
        """Submission 파일 생성"""

        results = []

        for eval_id, data in eval_data.items():
            messages = data.get('messages', [])

            # 최종 쿼리와 대화 이력 추출
            final_query = self.extract_final_query(messages)
            chat_history = self.extract_chat_history(messages)

            if not final_query:
                doc_ids = []
            else:
                # 검색 수행
                doc_ids = self.search_system.search_with_reranking(
                    final_query, chat_history
                )

            # JSON 문자열로 변환
            doc_ids_str = json.dumps(doc_ids, ensure_ascii=False)

            results.append({
                'evaluation_id': eval_id,
                'doc_ids': doc_ids_str
            })

            print(f"처리 완료: {eval_id} -> {len(doc_ids)}개 문서")

        # CSV 파일 저장
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"Submission 파일 생성 완료: {output_path}")
        return df


# 사용 예시
generator = SubmissionGenerator(search_system)

# 평가 데이터 로딩
eval_data = generator.load_evaluation_data("evaluation_data.json")

# Submission 파일 생성
submission_df = generator.generate_submission(eval_data)

print(submission_df.head())
```

### 2. 검증 및 디버깅

```python
class SubmissionValidator:
    def __init__(self):
        pass

    def validate_submission(self, submission_path: str) -> bool:
        """Submission 파일 유효성 검사"""
        try:
            df = pd.read_csv(submission_path)

            # 필수 컬럼 확인
            required_cols = ['evaluation_id', 'doc_ids']
            if not all(col in df.columns for col in required_cols):
                print("❌ 필수 컬럼 누락")
                return False

            # 각 행 검증
            for idx, row in df.iterrows():
                eval_id = row['evaluation_id']
                doc_ids_str = row['doc_ids']

                try:
                    # JSON 파싱 테스트
                    doc_ids = json.loads(doc_ids_str)

                    # 타입 검증
                    if not isinstance(doc_ids, list):
                        print(f"❌ {eval_id}: doc_ids가 리스트가 아님")
                        return False

                    # 길이 검증
                    if len(doc_ids) > 3:
                        print(f"❌ {eval_id}: 문서 개수가 3개 초과 ({len(doc_ids)}개)")
                        return False

                    # 문서 ID 형식 검증
                    for doc_id in doc_ids:
                        if not isinstance(doc_id, str) or len(doc_id) == 0:
                            print(f"❌ {eval_id}: 잘못된 문서 ID 형식")
                            return False

                except json.JSONDecodeError:
                    print(f"❌ {eval_id}: JSON 파싱 오류")
                    return False

            print("✅ Submission 파일 검증 통과")
            return True

        except Exception as e:
            print(f"❌ 검증 중 오류: {e}")
            return False

    def analyze_submission(self, submission_path: str):
        """Submission 통계 분석"""
        df = pd.read_csv(submission_path)

        doc_counts = []
        empty_count = 0

        for _, row in df.iterrows():
            doc_ids = json.loads(row['doc_ids'])
            doc_counts.append(len(doc_ids))
            if len(doc_ids) == 0:
                empty_count += 1

        print("=== Submission 분석 결과 ===")
        print(f"총 쿼리 수: {len(df)}")
        print(f"빈 결과 수: {empty_count} ({empty_count / len(df) * 100:.1f}%)")
        print(f"평균 문서 수: {sum(doc_counts) / len(doc_counts):.2f}")
        print(f"문서 수 분포:")
        for i in range(4):
            count = doc_counts.count(i)
            print(f"  {i}개: {count}개 ({count / len(doc_counts) * 100:.1f}%)")


# 사용 예시
validator = SubmissionValidator()

# 검증
is_valid = validator.validate_submission("submission.csv")

if is_valid:
    # 분석
    validator.analyze_submission("submission.csv")
```

---

## 🚀 고급 최적화 기법

### 1. 하이브리드 검색

```python
class HybridSearchSystem(IRSearchSystem):
    def __init__(self, vectorstore, sparse_retriever=None):
        super().__init__(vectorstore)
        self.sparse_retriever = sparse_retriever  # BM25 등

    def hybrid_search(self, query: str, chat_history: str = "") -> List[str]:
        """Dense + Sparse 하이브리드 검색"""

        analysis = self.analyzer.analyze_query(query)
        if not analysis["needs_search"]:
            return []

        standalone_query = self.analyzer.generate_standalone_query(
            query, chat_history
        )

        # Dense 검색 (Vector Store)
        dense_docs = self.vectorstore.similarity_search(standalone_query, k=8)

        # Sparse 검색 (BM25 등)
        if self.sparse_retriever:
            sparse_docs = self.sparse_retriever.get_relevant_documents(
                standalone_query
            )[:8]
        else:
            sparse_docs = []

        # 결과 병합 및 중복 제거
        combined_docs = dense_docs + sparse_docs
        seen_ids = set()
        unique_docs = []

        for doc in combined_docs:
            doc_id = doc.metadata.get('doc_id')
            if doc_id and doc_id not in seen_ids:
                unique_docs.append(doc)
                seen_ids.add(doc_id)

        # LLM 리랭킹
        reranked_docs = self._rerank_with_llm(query, unique_docs[:10])

        # Top-3 반환
        doc_ids = []
        for doc in reranked_docs[:3]:
            doc_id = doc.metadata.get('doc_id')
            if doc_id:
                doc_ids.append(doc_id)

        return doc_ids
```

### 2. 다중 쿼리 생성

```python
class MultiQuerySearchSystem(IRSearchSystem):
    def __init__(self, vectorstore):
        super().__init__(vectorstore)

        self.multi_query_prompt = ChatPromptTemplate.from_messages([
            ("system", """주어진 질문에 대해 3개의 다른 관점에서 검색 쿼리를 생성해주세요.
각 쿼리는 다른 키워드나 표현을 사용하여 동일한 정보를 찾을 수 있어야 합니다.

형식:
1. 첫 번째 쿼리
2. 두 번째 쿼리
3. 세 번째 쿼리"""),
            ("user", "{question}")
        ])

    def generate_multiple_queries(self, question: str) -> List[str]:
        """다중 쿼리 생성"""
        chain = self.multi_query_prompt | self.llm
        response = chain.invoke({"question": question})

        lines = response.content.strip().split('\n')
        queries = []

        for line in lines:
            if line.strip() and any(line.startswith(str(i)) for i in range(1, 4)):
                query = line.split('.', 1)[1].strip()
                queries.append(query)

        return queries

    def multi_query_search(self, query: str, chat_history: str = "") -> List[str]:
        """다중 쿼리를 사용한 검색"""

        analysis = self.analyzer.analyze_query(query)
        if not analysis["needs_search"]:
            return []

        # 다중 쿼리 생성
        queries = self.generate_multiple_queries(query)

        # 각 쿼리로 검색
        all_docs = []
        for q in queries:
            docs = self.vectorstore.similarity_search(q, k=5)
            all_docs.extend(docs)

        # 중복 제거 및 빈도 기반 점수
        doc_scores = {}
        for doc in all_docs:
            doc_id = doc.metadata.get('doc_id')
            if doc_id:
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1

        # 빈도 순으로 정렬
        sorted_docs = sorted(doc_scores.items(),
                             key=lambda x: x[1], reverse=True)

        return [doc_id for doc_id, _ in sorted_docs[:3]]
```

---

## 🎯 완전한 실행 파이프라인

```python
def main():
    """전체 파이프라인 실행"""

    print("=== IR 경진대회 Submission 생성 파이프라인 ===")

    # 1. 문서 로딩
    print("\n1. 문서 데이터 로딩...")
    loader = DocumentLoader("documents.csv")
    documents = loader.load_documents()

    # 2. 문서 청킹
    print("\n2. 문서 청킹...")
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    chunked_documents = chunker.chunk_documents(documents)

    # 3. Vector Store 구축
    print("\n3. Vector Store 구축...")
    builder = VectorStoreBuilder()
    vectorstore = builder.build_chroma_store(chunked_documents)

    # 4. 검색 시스템 초기화
    print("\n4. 검색 시스템 초기화...")
    search_system = IRSearchSystem(vectorstore, top_k=10)

    # 5. Submission 생성
    print("\n5. Submission 파일 생성...")
    generator = SubmissionGenerator(search_system)
    eval_data = generator.load_evaluation_data("evaluation_data.json")
    submission_df = generator.generate_submission(eval_data)

    # 6. 검증
    print("\n6. 파일 검증...")
    validator = SubmissionValidator()
    is_valid = validator.validate_submission("submission.csv")

    if is_valid:
        validator.analyze_submission("submission.csv")
        print("\n✅ 파이프라인 완료! submission.csv 파일이 생성되었습니다.")
    else:
        print("\n❌ 검증 실패. 파일을 확인해주세요.")


if __name__ == "__main__":
    main()
```

---

## 💡 성능 최적화 팁

### 1. 임베딩 모델 선택

```python
# 다양한 임베딩 모델 시도
embedding_options = [
    "text-embedding-3-small",  # 빠르고 저렴
    "text-embedding-3-large",  # 더 높은 성능
    "text-embedding-ada-002"  # 기본 모델
]
```

### 2. 청킹 전략 최적화

```python
# 과학 문서 특성에 맞는 청킹
chunker = DocumentChunker(
    chunk_size=800,  # 과학 개념 설명에 충분한 크기
    chunk_overlap=100,  # 적절한 오버랩
    separators=["\n\n", "\n", ". ", " "]  # 과학 문서 구조 고려
)
```

### 3. 검색 매개변수 튜닝

```python
# 다양한 검색 설정 실험
search_configs = [
    {"k": 5, "score_threshold": 0.7},
    {"k": 10, "score_threshold": 0.6},
    {"k": 15, "score_threshold": 0.5}
]
```
