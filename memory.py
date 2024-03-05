import os
import openai
import datetime
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory
)

class LangChainManager:
    def __init__(self):
        load_dotenv(find_dotenv())
        openai.api_key = os.environ['OPENAI_API_KEY']
        self.current_date = datetime.datetime.now().date()
        self.target_date = datetime.date(2024, 6, 12)
        self.llm_model = self.get_llm_model()

    def get_llm_model(self):
        return "gpt-3.5-turbo" if self.current_date > self.target_date else "gpt-3.5-turbo-0301"

class MemoryDemo:
    @staticmethod
    def demonstrate_conversation_buffer_memory(llm_model):
        memory = ConversationBufferMemory()
        memory.save_context({"input": "Hi"}, {"output": "What's up"})
        memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
        print(f"Memory variables ConversationBufferMemory: {memory.load_memory_variables({})}\n")

    @staticmethod
    def demonstrate_conversation_buffer_window_memory(llm_model, k=1):
        memory = ConversationBufferWindowMemory(k=k)
        memory.save_context({"input": "Hi"}, {"output": "What's up"})
        memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
        print(f"Memory variables ConversationBufferWindowMemory: {memory.load_memory_variables({})}\n")

    @staticmethod
    def demonstrate_conversation_token_buffer_memory(llm_model, max_token_limit=30):
        llm = ChatOpenAI(temperature=0.0, model=llm_model)
        memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=max_token_limit)
        memory.save_context({"input": "AI is what?!"}, {"output": "Amazing!"})
        memory.save_context({"input": "Backpropagation is what?"}, {"output": "Beautiful!"})
        memory.save_context({"input": "Chatbots are what?"}, {"output": "Charming!"})
        print(f"Memory variables ConversationTokenBufferMemory: {memory.load_memory_variables({})}\n")

    @staticmethod
    def demonstrate_conversation_summary_buffer_memory(llm_model, max_token_limit=100):
        llm = ChatOpenAI(temperature=0.0, model=llm_model)
        schedule = "There is a meeting at 8am with your product team. You will need your powerpoint presentation prepared. 9am-12pm have time to work on your LangChain project which will go quickly because Langchain is such a powerful tool. At Noon, lunch at the italian resturant with a customer who is driving from over an hour away to meet you to understand the latest in AI. Be sure to bring your laptop to show the latest LLM demo."
        memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=max_token_limit)
        memory.save_context({"input": "Hello"}, {"output": "What's up"})
        memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
        #memory.save_context({"input": "What is on the schedule today?"}, {"output": f"{schedule}"})
        print(f"Memory variables ConversationSummaryBufferMemory: {memory.load_memory_variables({})}\n")

if __name__ == "__main__":
    lang_chain_manager = LangChainManager()
    print(f"Lang Chain Manager: {lang_chain_manager}")

    MemoryDemo.demonstrate_conversation_buffer_memory(lang_chain_manager.llm_model)
    MemoryDemo.demonstrate_conversation_buffer_window_memory(lang_chain_manager.llm_model)
    MemoryDemo.demonstrate_conversation_token_buffer_memory(lang_chain_manager.llm_model)
    MemoryDemo.demonstrate_conversation_summary_buffer_memory(lang_chain_manager.llm_model)
