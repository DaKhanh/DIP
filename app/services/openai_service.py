# /services/openai_service.py
import openai
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

class OpenAIService:
    def __init__(self):
        # 从环境变量中获取 API 密钥
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.llm = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-3.5-turbo", temperature=0, max_tokens=1400)

    def generate_response(self, context, chat_history):
        # 与之前相同的生成回答逻辑
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
