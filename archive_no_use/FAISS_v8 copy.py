
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
        context = f"Instruction: You are a helpful assistant
          designed to help NTU students find courses.
            Provide accurate information about available courses at
              Nanyang 
              Technological University.\n\nDocuments: {doc}\n\nQuestion: {question}"

        # Set up QA chain for course-related question
        llm_chain = setup_llm_chain()
        response = llm_chain.invoke({"context": context})
        return response
    else:
        # For non-course-related questions, use GPT to answer directly
        context = f"Instruction: You are a general-purpose assistant.
          Answer the following question accurately and helpfully.\n\nQuestion: {question}"
        llm_chain = setup_llm_chain()
        response = llm_chain.invoke({"context": context})
        return response

# %%
response = chat_with_llm_chain(question)
print(response['text'])

# %%



