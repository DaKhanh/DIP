# FAISS_v8.py
# %%
!pip install -U langchain-openai
!pip install python-dotenv
!pip install python-dotenv langchain openai faiss-cpu
!pip install langchain openai faiss-cpu
!pip install --upgrade langchain
!pip install langchain_community
!pip install langchain_openai

# %% [markdown]
# # New section

# %%
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.chat_models import ChatOpenAI
from google.colab import userdata

# %%
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# %%
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# %%


# %%
os.environ.pop('OPENAI_API_KEY', None)


openai_api_key = 'sk-ebdM8qcZ6KODTqhglzCiTM1hRWg9JgIHNWVhpsqbg4T3BlbkFJErmk6gObezeGCLdHfZ8aMbJsxw6qx1tukdniAtY2cA'

print(f"OpenAI API Key: {openai_api_key}")

if not openai_api_key or not openai_api_key.startswith('sk-'):
    raise ValueError("OpenAI API key is missing or invalid")

# %%


# %%
#data = pd.read_excel(r'C:\Users\cheww\Documents\Y3S1\DIP Project\modsoptimizer.xlsx')

# %%
file_path = (r"/content/modsoptimizerv3.csv")
#df = pd.read_csv(file_path)
#loader = [Document(page_content=row.to_string()) for _, row in df.iterrows()]
loader = CSVLoader(file_path=file_path, csv_args={"delimiter": ",",})
data = loader.load()
print(len(data))

# %%
#data_text = ""
#for index, row in data.iterrows():
#    row_text = " | ".join([f"{col}: {row[col]}" for col in data.columns])
#    data_text += row_text + "\n"

# %%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
texts = text_splitter.split_documents(data)

# %%
texts_for_faiss = [doc.page_content for doc in texts]
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

# %%
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
vector_store.add_documents(documents=texts)

# %%
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo",
    temperature=0,  # Adjust for creative vs factual answer
    max_tokens = 1400
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# %% [markdown]
# # New section

# %%
question = "what is the best eee module"

# %%
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_keywords(question):
    doc = nlp(question)
    keywords = []
    unwanted_words = {"course", "code", "name", "description", "list"}
    # Extract nouns and proper nouns as keywords, excluding unwanted words
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and token.text.lower() not in unwanted_words:
            keywords.append(token.text)

    # Remove duplicates by converting the list to a set
    unique_keywords = set(keywords)
    return " ".join(unique_keywords)
keywords = extract_keywords(question)

print(keywords)


def is_course_related(question):
    course_keywords = ["introduce","prereq", "prerequisite", "AU", "details", "course", "recommend", "subject", "class", "module", "NTU", "major", "elective", "learn","mod", "BDE", ""]
    doc = nlp(question.lower())

    # Check if any of the course-related keywords are present in the question
    for token in doc:
        if token.text in course_keywords:
            return True
    return False

# %%
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


        """
    )

    llm_chain = LLMChain(
        prompt=prompt_template,
        llm=llm,
        memory=memory
    )
    return llm_chain

# %%
def chat_with_llm_chain(question):
    # Classify if the question is related to course
    keywords = extract_keywords(question)
    retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={'k': 50}
    )
    doc = retriever.invoke(keywords)
    if is_course_related(question):
        # Create an instruction for course recommendation scenario
        context = f"Instruction: You are a helpful assistant designed to help NTU students find courses. Provide accurate information about available courses at Nanyang Technological University.\n\nDocuments: {doc}\n\nQuestion: {question}"

        # Set up QA chain for course-related question
        llm_chain = setup_llm_chain()
        response = llm_chain.invoke({"context": context})
        return response
    else:
        # For non-course-related questions, use GPT to answer directly
        context = f"Instruction: You are a general-purpose assistant. Answer the following question accurately and helpfully.\n\nQuestion: {question}"
        llm_chain = setup_llm_chain()
        response = llm_chain.invoke({"context": context})
        return response

# %%
response = chat_with_llm_chain(question)
print(response['text'])

# %%



