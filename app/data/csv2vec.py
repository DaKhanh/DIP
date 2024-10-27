import os
import faiss
import pickle
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Missing OpenAI API key")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

csv_path = 'D:\\dip_all\\app\\data\\data_cleaned.csv'
loader = CSVLoader(file_path=csv_path, csv_args={"delimiter": ","})
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
texts = text_splitter.split_documents(data)

texts_for_faiss = [doc.page_content for doc in texts]

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(documents=texts)

# 存储向量索引和数据到磁盘
faiss.write_index(index, "faiss_index.index")
with open("faiss_docstore.pkl", "wb") as f:
    pickle.dump(vector_store.docstore, f)
with open("faiss_id_map.pkl", "wb") as f:
    pickle.dump(vector_store.index_to_docstore_id, f)

print(f"FAISS index and documents saved to disk. Number of documents: {index.ntotal}")
