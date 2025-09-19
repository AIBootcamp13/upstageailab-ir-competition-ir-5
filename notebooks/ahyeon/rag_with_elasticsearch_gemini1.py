import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import traceback
import time


# ==============================
# 1. Sentence Transformer 모델 초기화
# ==============================
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

def get_embedding(sentences):
    return model.encode(sentences)

def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings


# ==============================
# 2. Elasticsearch 설정
# ==============================
def create_es_index(index, settings, mappings):
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, settings=settings, mappings=mappings)

def delete_es_index(index):
    es.indices.delete(index=index)

def bulk_add(index, docs):
    actions = [{"_index": index, "_source": doc} for doc in docs]
    return helpers.bulk(es, actions)

def sparse_retrieve(query_str, size):
    query = {"match": {"content": {"query": query_str}}}
    return es.search(index="test", query=query, size=size, sort="_score")

def dense_retrieve(query_str, size):
    query_embedding = get_embedding([query_str])[0]
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }
    return es.search(index="test", knn=knn)



es_username = "elastic"
es_password = "q+t6xDTHzKPFGA85S9kX"

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="../elasticsearch-8.8.0/config/certs/http_ca.crt")

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



# ==============================
# 3. Gemini 설정
# ==============================
genai.configure(api_key="your_api_key")  # 실제 API 키로 교체
llm_model = "gemini-2.5-flash"

persona_qa = """
## Role: 과학 상식 전문가
## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

def gemini_chat(messages, system_prompt=""):
    model = genai.GenerativeModel(llm_model)

    # messages를 하나의 문자열로 합치기
    prompt = system_prompt + "\n\n"
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "user":
            prompt += f"사용자: {content}\n"
        elif role == "assistant":
            prompt += f"AI: {content}\n"
        else:  # system
            prompt += f"{content}\n"

    response = model.generate_content(prompt)
    return response.text



# ==============================
# 4. RAG 질의응답
# ==============================
def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    try:
        # 마지막 메시지를 질의로 간주
        user_query = messages[-1]["content"]

        # 우선 sparse 검색 실행
        search_result = sparse_retrieve(user_query, 3)
        
        retrieved_context = []
        for rst in search_result['hits']['hits']:
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"].get("docid", ""))
            response["references"].append({
                "score": rst["_score"],
                "content": rst["_source"]["content"]
            })

        # Gemini에 질의 + 검색 결과 전달
        prompt = persona_qa + "\n\n"
        prompt += f"질문: {user_query}\n"
        prompt += f"참고 문서: {retrieved_context}\n"

        answer = gemini_chat([{"role": "user", "content": prompt}])
        response["answer"] = answer

    except Exception as e:
        traceback.print_exc()
        response["answer"] = "오류가 발생했습니다."

    return response



# ==============================
# 5. 평가 실행
# ==============================
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

            if idx % 1 == 0:  # 매 호출마다 대기
                print("Waiting for 40 seconds to avoid rate limit...")
                time.sleep(40) # 40초 대기

eval_rag("../korea202/data/eval.jsonl", "./data/sample_submission.csv")