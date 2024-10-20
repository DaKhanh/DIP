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
from utils.helper_functions import extract_keywords, is_course_related
from dotenv import load_dotenv

# 加载环境变量中的 API Key
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("Missing OpenAI API key")

# 初始化 OpenAI 的嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

# 加载CSV文件并创建文档
csv_path = 'data/modsoptimizerv3.csv'
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

# 确保索引成功添加了文档
print(f"Number of documents in FAISS index: {index.ntotal}")

# 初始化 ChatOpenAI 语言模型
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo",
    temperature=0,  # 控制回答的创造性
    max_tokens=1400  # 控制每个回答的最大长度
)

# 设置 LLMChain 的函数
def setup_llm_chain():
    """设置 LLMChain 的 Prompt 模板和链条"""
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
    """使用 FAISS 和 LLM 检索并回答用户问题"""
    # 使用正则化的关键词提取
    keywords = extract_keywords(question)
    print(f"Extracted keywords: {keywords}")  # 调试输出

    # 调用 FAISS 检索器
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 10})
    doc = retriever.invoke(keywords)
    
    print(f"Retrieved documents: {doc}")  # 调试输出
    if not doc:
        print("No documents retrieved, FAISS retrieval might not be working correctly.")


    # 根据问题是否与课程相关，调用不同的回答策略
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
