import os
import faiss
import pickle
import pandas as pd
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Missing OpenAI API key")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

csv_path = 'D:\\dip_all\\app\\data\\data_cleaned.csv'

df = pd.read_csv(csv_path, index_col=0)

documents = []

for idx, row in df.iterrows():
    row_dict = row.to_dict()
    row_text = json.dumps(row_dict, ensure_ascii=False)
    doc = Document(page_content=row_text)
    documents.append(doc)

print("example:")
for i, doc in enumerate(documents[:3]):  
    print(f"\n--- doc {i+1} ---")
    print(doc.page_content)
    print(f"length: {len(doc.page_content)} characters")


texts_for_faiss = [doc.page_content for doc in documents]

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(documents=documents)

# 存储向量索引和数据到磁盘
faiss.write_index(index, "faiss_index.index")
with open("faiss_docstore.pkl", "wb") as f:
    pickle.dump(vector_store.docstore, f)
with open("faiss_id_map.pkl", "wb") as f:
    pickle.dump(vector_store.index_to_docstore_id, f)

print(f"FAISS index and documents saved to disk. Number of documents: {index.ntotal}")