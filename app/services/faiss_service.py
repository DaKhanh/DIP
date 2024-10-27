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
from flashrank import Ranker, RerankRequest
import pickle
import json
import time


load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Missing OpenAI API key")

index = faiss.read_index("D:\\dip_all\\app\\data\\faiss_index.index")
with open("D:\\dip_all\\app\\data\\faiss_docstore.pkl", "rb") as f:
    docstore = pickle.load(f)
with open("D:\\dip_all\\app\\data\\faiss_id_map.pkl", "rb") as f:
    index_to_docstore_id = pickle.load(f)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)

print(f"Loaded FAISS index with {index.ntotal} documents.")

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo",
    temperature=0,  
    max_tokens=1400  
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def setup_llm_chain():
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "context"],
        template="""
        You are an expert assistant. Here's the conversation so far:
        {chat_history}
        Now, use the following context to answer the question:
        {context}
        Provide a helpful and accurate answer.

        Here some more information you need to consider regarding column provided in the data:

        Core: module that must be taken by the major
        BDE is Broadening deepening electives. These are the module that is available to students outside of their core to be taken.

        A student can not take a BDE from a module that are ran from their department. For instance, if you are a EEE student you can not take EE3101 as a BDE
        Finally, if you are asked about details regarding a certain module please provide the course code, description, academic units, course title and prerequisite and dont include level

        If you are given questions that is related to subjective judgements, please provide a disclaimer that you dont have the exact data to backup your statement. such as when you are asked about which is the best mod

        """,
    )
    
    llm_chain = LLMChain(
        prompt=prompt_template,
        llm=llm,
        memory=memory
    )
    return llm_chain

def chat_with_llm_chain(question):
    keywords = extract_keywords(question)
    print(f"Extracted keywords: {keywords}")  

    retriever = vector_store.as_retriever(search_type="similarity", 
                                          search_kwargs={'k': 100})
    initial_docs = retriever.invoke(keywords)
    
    # print(f"Retrieved documents: {doc}")  
    if not initial_docs:
        print("No documents retrieved, FAISS retrieval might not be working correctly.")
    
    start_time = time.time()
    # prepare the initial results in the format expected by FlashRank
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(initial_docs)]
    print(f"Retrieval time: {time.time() - start_time}")

    # save the initial resultsc as a json file at "D:\\dip_all\\app\\data\\initial_results.json"
    initial_results_path = "D:\\dip_all\\app\\data\\initial_results.json"
    with open(initial_results_path, 'w') as json_file:
        json.dump(passages, json_file, indent=4)
    print(f"Saved JSON file to {initial_results_path}")
    
    print(f"Write time: {time.time() - start_time}")

    # FlashRank for re-ranking
    ############################# Reranking ################################
    ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")  
    rerank_request = RerankRequest(query=question, passages=passages)
    reranked_docs = ranker.rerank(rerank_request)
    print(f"Reranking time: {time.time() - start_time}")

    if not reranked_docs or len(reranked_docs) == 0:
        print("No documents passed the refinement stage.")
        return None
    print(f"Reranked documents: {len(reranked_docs)}")
    

    # save the reranked results at "D:\\dip_all\\app\\data\\reranked_results.json"
    # remove the 'score' field from the document
    for doc in reranked_docs:
        if 'score' in doc:
            del doc['score']

    reranked_results_path = "D:\\dip_all\\app\\data\\reranked_results.json"
    with open(reranked_results_path, 'w') as json_file:
        json.dump(reranked_docs, json_file, indent=4)
    print(f"Saved JSON file to {reranked_results_path}")

    # limit the number of documents to 10 to avoid long prompts
    top_10_reranked_docs = reranked_docs[:10]
    
    context_documents = "\n\n".join([doc["text"] for doc in top_10_reranked_docs])
    print(f"join time: {time.time() - start_time}")
    
    # construct prompt
    # if is_course_related(question):
    #     context = f"Documents: {context_documents}\n\nQuestion: {question}"
    # else:
    #     context = f"General question: {question}"
    
    context = f"Documents: {context_documents}\n\nQuestion: {question}"

    # save the prompt to a file
    prompt_file_path = "D:\\dip_all\\app\\data\\prompt.txt"
    with open(prompt_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Prompt:\n{context}\n")
    print(f"Saved prompt to {prompt_file_path}")

    # LLMChain
    llm_chain = setup_llm_chain()
    response = llm_chain.invoke({
        "context": context, 
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })

    # save the LLM response to a file
    response_file_path = "D:\\dip_all\\app\\data\\llm_response.txt"
    with open(response_file_path, 'w', encoding='utf-8') as f:
        f.write(f"LLM Response:\n{response}\n")
    print(f"Saved LLM response to {response_file_path}")
        
    return response
