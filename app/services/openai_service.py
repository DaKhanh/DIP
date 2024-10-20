# /services/openai_service.py
import openai
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI

load_dotenv()

class OpenAIService:
    def __init__(self):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.llm = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-3.5-turbo", temperature=0, max_tokens=1400)

    def generate_response(self, context, chat_history):
        prompt_template = PromptTemplate(
            input_variables=["chat_history", "context"],
            template="""
            You are an expert assistant. Here's the conversation so far:
            {chat_history}
            Now, use the following context to answer the question:
            {context}
            Provide a helpful and accurate answer.
            """
        )
        llm_chain = LLMChain(prompt=prompt_template, llm=self.llm)
        return llm_chain.run({"context": context, "chat_history": chat_history})
