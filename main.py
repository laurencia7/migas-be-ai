import os
import psycopg2
import requests
from fastapi import FastAPI, Form
import json
from ollama import Client
import cohere

# --------------------------------------------------
#  FASTAPI APP
# --------------------------------------------------
app = FastAPI()
HOST     = os.getenv("HOST")
client = Client(HOST)
# --------------------------------------------------
#  CONFIG DB
# --------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

DBNAME     = os.getenv("DBNAME")
DBUSER     = os.getenv("DBUSER")
DBPASSWORD = os.getenv("DBPASSWORD")
DBHOST     = os.getenv("DBHOST")
DBPORT     = os.getenv("DBPORT")
RERANKER_MODEL = os.getenv("RERANKER_MODEL")
COHERE_API = os.getenv("COHERE_API")

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
co = cohere.ClientV2(COHERE_API)

# --------------------------------------------------
#  GET CONTEXT FROM POSTGRES
# --------------------------------------------------
def get_context(conversation_id: str) -> str:
    try:
        conn = psycopg2.connect(
            dbname=DBNAME,
            user=DBUSER,
            password=DBPASSWORD,
            host=DBHOST,
            port=DBPORT
        )
        cursor = conn.cursor()

        cursor.execute("""
            SELECT query, answer 
            FROM validation_history 
            WHERE conversation_id = %s
            ORDER BY created_at ASC
        """, (conversation_id,))

        rows = cursor.fetchall()

        if not rows:
            return ""

        parts = []
        for q, a in rows:
            parts.append(f"User: {q}\nJawaban: {a}\n")

        return "\n".join(parts)

    except Exception as e:
        print("Error get_context:", e)
        return ""
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()


def rewrite_query(query: str, context: str) -> str:
    prompt = f"""
Your role is to act as an expert query rewriter.
Your task is to rewrite the given <user_query> into a more a precise, specific, complete, and effective search query based on the provided <context>.

RULES:
1. If the <user_query> is just a greeting (like "halo", "hi", etc.), return the query as is or return the main topic from the context.
2. DO NOT answer the user. DO NOT engage in conversation.
3. ONLY output the rewritten query text.
4. If no rewrite is needed, output the original <user_query>.
5. Language: Bahasa Indonesia.
6. PRESERVE TECHNICAL TERMS: Do not remove or over-abbreviate key technical terms (e.g., keep "General Overhaul" instead of just "GOH" if the full term adds clarity). Precision is more important than brevity.

<context>
{context}
</context>

<user_query>
{query}
</user_query>
    """

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt}
        ],
        "stream": False
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()

        data = resp.json()

        # Format response Ollama
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"].strip()

        return ""

    except Exception as e:
        print("Error rewrite:", e)
        return ""

# --------------------------------------------------
#  API ENDPOINT — RETURN STRING PLAIN
# --------------------------------------------------

# def rerank_document(query: str, document: str) -> float:
#     prompt = f"""
#     You are an expert relevance grader. Is the following document relevant to the user's query? Reply 'Yes' or 'No'.
#     Query: {query}
#     Document: {document}
#     """
#     try:
#         response = client.chat(
#             model=RERANKER_MODEL,
#             messages=[{'role': 'user', 'content': prompt}],
#             options={'temperature': 0.0}
#         )
#         answer = response['message']['content'].strip().lower()
#         return 1.0 if 'yes' in answer else 0.0
#     except Exception as e:
#         print(f"Error: {e}")
#         return 0.0
    
def get_relevance_score(query: str, doc: str) -> float:
    prompt = f"""You are an expert relevance grader. Score the relevance of the document to the query on a scale of 0.0 to 1.0.
    1.0 is highly relevant, 0.0 is completely irrelevant.
    Query: {query}
    Document: {doc}
    Just reply with the score."""
    
    try:
        response = client.chat(
            model=RERANKER_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0}
        )
        content = response['message']['content'].strip()
        import re
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)
        return float(numbers[0]) if numbers else 0.0
    except Exception as e:
        print(f"Rerank error for doc: {e}")
        return 0.0
    #     # Ambil angka saja dari output
    #     score = float(''.join(c for c in content if c.isdigit() or c == '.'))
    #     return score
    # except:
    #     return 0.0
    

@app.post("/rerank-ollama")
def rerank(
    query: str = Form(...),
    documents: str = Form(...) 
):
    print("masuk rerank")
    try:
        doc_list = json.loads(documents)
    except Exception as e:
        return {"error": "Invalid JSON format", "details": str(e)}
        
    scored_docs = []
    for doc in doc_list:
        text = doc.get('content', '')
        print("before hit model")
        score = get_relevance_score(query, text)
        scored_docs.append({"doc": doc, "score": score})
    
    # Sort descending
    scored_docs.sort(key=lambda x: x['score'], reverse=True)
    
    # Ambil Top 3
    top_3 = [item['doc'] for item in scored_docs[:3]]
    
    # Kembalikan sebagai object agar Dify mudah memprosesnya
    return {"results": top_3}

@app.post("/rerank")
def rerank(
    query: str = Form(...),
    documents: str = Form(...),
    is_qna: str = Form("false")
):
    print(f"Masuk rerank. Mode QnA: {is_qna}")
    try:
        doc_list = json.loads(documents)

        texts = [doc.get('content', '') for doc in doc_list]
        
        if not texts:
            return {"results": []}

        is_qna_bool = is_qna.lower() == "true"
        target_top_k = 1 if is_qna_bool else 3

        response = co.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=texts,
            top_n=target_top_k,
        )

        top_k = []
        for result in response.results:
            # result.index adalah urutan index dokumen asli yang dikirim
            original_doc = doc_list[result.index]
            # Tambahkan skor dari cohere ke metadata (opsional)
            original_doc['rerank_score'] = result.relevance_score
            top_k.append(original_doc)

        print(f"Rerank selesai. Berhasil memproses {len(texts)} dokumen.")
        print("top k")
        print(top_k)
        return {"results": top_k}

    except Exception as e:
        print(f"Cohere Rerank Error: {e}")
        # Fallback: jika API error, kirim 3 dokumen pertama agar flow tidak putus
        try:
            return {"results": json.loads(documents)[:3]}
        except:
            return {"results": []}
            
        
@app.post("/rewrite")
def rewrite(
    conversation_id: str = Form(...),
    query: str = Form(...)
):
    print("conversation_id =", repr(conversation_id))
    print("query =", repr(query))

    context = get_context(conversation_id)
    rewritten = rewrite_query(query, context)
    print("rewritten: ", rewritten)

    return rewritten
