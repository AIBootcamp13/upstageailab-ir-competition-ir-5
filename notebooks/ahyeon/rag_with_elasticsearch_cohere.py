import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import time
import cohere
import traceback

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
model = SentenceTransformer("monologg/kobert-base-v2")


# SentenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    return model.encode(sentences)


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings


# 새로운 index 생성
def create_es_index(index, settings, mappings):
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, settings=settings, mappings=mappings)


# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)


# 역색인을 이용한 검색
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")


# Vector 유사도를 이용한 검색
def dense_retrieve(query_str, size):
    query_embedding = get_embedding([query_str])[0]
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }
    return es.search(index="test", knn=knn)


# Elasticsearch 연결 설정
es_username = "elastic"
es_password = "q+t6xDTHzKPFGA85S9kX"

es = Elasticsearch(
    ['https://localhost:9200'],
    basic_auth=(es_username, es_password),
    ca_certs="../elasticsearch-8.8.0/config/certs/http_ca.crt"
)

print(es.info())

settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "l2_norm"
        }
    }
}

create_es_index("test", settings, mappings)

index_docs = []
with open("../korea202/data/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)

for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

ret = bulk_add("test", index_docs)
print(ret)

# -------------------------------
# Cohere API 기반 RAG 코드
# -------------------------------

# Cohere API 키 설정
os.environ["COHERE_API_KEY"] = "your_api_key"  # 실제 API 키로 교체
co = cohere.Client(os.environ["COHERE_API_KEY"])
llm_model = "command-r-plus"

persona_qa = """
## Role: 과학 상식 전문가
## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

persona_function_calling = """
## Role: 과학 상식 전문가
## Instruction
- 사용자가 대화를 통해 과학 지식에 관한 질문을 하면 반드시 검색 쿼리를 만들어 반환한다.
- 과학 상식과 관련되지 않은 질문이라면 반드시 "NONE" 이라고만 반환한다.
"""


def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    msg_text = persona_function_calling + "\n" + "\n".join(
        [f"{m['role']}: {m['content']}" for m in messages]
    )

    try:
        result = co.chat(
            model=llm_model,
            message=msg_text,
            temperature=0,
        )
    except Exception:
        traceback.print_exc()
        return response

    standalone_query = result.text.strip()

    # 과학 상식 질문일 때만 검색 → RAG 수행
    if standalone_query and standalone_query != "NONE":
        search_result = sparse_retrieve(standalone_query, 3)
        response["standalone_query"] = standalone_query
        retrieved_context = []
        for rst in search_result['hits']['hits']:
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"].get("docid", ""))
            response["references"].append({
                "score": rst["_score"],
                "content": rst["_source"]["content"]
            })

        context_text = "\n".join(retrieved_context)
        qa_prompt = persona_qa + f"\n\n사용자 질문: {standalone_query}\n\n참고 자료:\n{context_text}"

        try:
            qaresult = co.generate(
                model=llm_model,
                prompt=qa_prompt,
                temperature=0,
                max_tokens=500
            )
            response["answer"] = qaresult.generations[0].text.strip()
        except Exception:
            traceback.print_exc()
            return response
    else:
        # 과학 상식이 아닌 질문 → 그냥 LLM 답변
        try:
            direct_answer = co.chat(
                model=llm_model,
                message="\n".join([f"{m['role']}: {m['content']}" for m in messages]),
                temperature=0,
            )
            response["answer"] = direct_answer.text.strip()
        except Exception:
            traceback.print_exc()

    return response


def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            output = {
                "eval_id": j["eval_id"],
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"]
            }
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

            if idx % 1 == 0:
                print("Waiting for 4 seconds to avoid rate limit...")
                time.sleep(4)


# 실행 예시
eval_rag("../korea202/data/eval.jsonl", "./data/sample_submission.csv")
