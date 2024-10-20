import os
import faiss
from langchain_openai.embeddings import OpenAIEmbeddings  # 更新后的导入
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.helper_functions import extract_keywords, is_course_related
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("Missing OpenAI API key")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

csv_path = 'D:\\dip_all\\app\\data\\modsoptimizerv3.csv'
loader = CSVLoader(file_path=csv_path, csv_args={"delimiter": ","})
data = loader.load()
print(len(data))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
texts = text_splitter.split_documents(data)

texts_for_faiss = [doc.page_content for doc in texts]
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
vector_store.add_documents(documents=texts)

print(f"Number of documents in FAISS index: {index.ntotal}")

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo",
    temperature=0,  
    max_tokens=1400  
)

def setup_llm_chain():
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "context"],
        template="""
        Here's the conversation so far:
        {chat_history}
        Now, use the following context to answer the question:
        {context}
        """,
    )
    
    llm_chain = LLMChain(
        prompt=prompt_template,
        llm=llm,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )
    return llm_chain

def chat_with_llm_chain(question):
    keywords = extract_keywords(question)
    print(f"Extracted keywords: {keywords}")  

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 10})
    doc = retriever.invoke(keywords)
    
    print(f"Retrieved documents: {doc}")  
    if not doc:
        print("No documents retrieved, FAISS retrieval might not be working correctly.")


    if is_course_related(question):
        context = f"Documents: {doc}\n\nQuestion: {question}"
        llm_chain = setup_llm_chain()
        response = llm_chain.invoke({"context": context})
        return response
    else:
        context = f"General question: {question}"
        llm_chain = setup_llm_chain()
        response = llm_chain.invoke({"context": context})
        return response
