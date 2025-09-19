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

# 인덱스 생성 및 문서 추가 로직은 동일
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
- 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 search api를 호출할 수 있어야 한다.
- 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성한다.
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
# 4. RAG 질의응답 (수정된 부분)
# ==============================
def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    try:
        # Step 1: LLM을 활용하여 대화 맥락을 고려한 독립형 질의 생성
        # 이전 대화와 마지막 질문을 모두 전달하여 독립적인 질문으로 만듦
        conversation_history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        standalone_prompt = f"""
        다음 대화에서 마지막 사용자의 발화를 독립적인 질문으로 재작성하세요.
        대화:
        {conversation_history}

        독립적인 질문:
        """
        
        standalone_query = gemini_chat([{"role": "user", "content": standalone_prompt}])
        response["standalone_query"] = standalone_query.strip()
        print(f"Standalone Query: {response['standalone_query']}")

        # Step 2: 생성된 독립형 질의로 검색
        search_result = sparse_retrieve(response["standalone_query"], 3)

        retrieved_context = []
        for rst in search_result['hits']['hits']:
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"].get("docid", ""))
            response["references"].append({
                "score": rst["_score"],
                "content": rst["_source"]["content"]
            })

        # Step 3: LLM에 전체 대화 맥락과 검색 결과 전달하여 답변 생성
        full_prompt = persona_qa + "\n\n"
        full_prompt += f"대화 내용: {conversation_history}\n"
        full_prompt += f"참고 문서: {retrieved_context}\n"
        full_prompt += f"사용자: {messages[-1]['content']}\n"
        
        answer = gemini_chat([{"role": "user", "content": full_prompt}])
        response["answer"] = answer

    except Exception as e:
        traceback.print_exc()
        response["answer"] = "오류가 발생했습니다."

    return response

# ==============================
# 5. 평가 실행 (수정된 부분)
# ==============================
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            
            # j["msg"] 전체를 answer_question에 전달
            messages = j["msg"]
            
            print(f'Test {idx}\nQuestion: {messages[-1]["content"]}')
            response = answer_question(messages) # 전체 메시지 리스트를 전달
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